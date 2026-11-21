import math
import sys
import os
import shutil

import torch
import numpy as np
import cv2

import util.misc as misc
import util.lr_sched as lr_sched
import torch_fidelity
from checkpoint_io import use_ema
from resource_paths import fid_stats_path


def _cache_local_inception_weights(fid_statistics_file):
    """优先把共享 fid_stats 中的 Inception 权重接入 torch hub 缓存。"""
    fid_stats = os.path.abspath(fid_statistics_file)
    source = os.path.join(os.path.dirname(fid_stats), "weights-inception-2015-12-05-6726825d.pth")
    if not os.path.isfile(source):
        return

    # torch-fidelity 内部通过 torch.hub.load_state_dict_from_url 加载固定文件名；
    # 本机已经有共享权重时建立软链接，避免评估阶段访问外网。
    cache_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
    os.makedirs(cache_dir, exist_ok=True)
    target = os.path.join(cache_dir, os.path.basename(source))
    if os.path.isfile(target):
        return
    try:
        os.symlink(source, target)
    except FileExistsError:
        pass
    except OSError:
        # 某些远端文件系统不允许软链接，退化为一次本地复制。
        shutil.copy2(source, target)


def train_one_epoch(model, model_without_ddp, data_loader, optimizer, device, epoch, log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (x, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # normalize image to [-1, 1]
        x = x.to(device, non_blocking=True).to(torch.float32).div_(255)
        x = x * 2.0 - 1.0
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            loss, loss_dict = model(x, labels)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if device.type == 'cuda':
            torch.cuda.synchronize()

        model_without_ddp.update_ema()

        metric_logger.update(loss=loss_value, **loss_dict)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        metrics_reduce = {key: misc.all_reduce_mean(value) for key, value in loss_dict.items()}

        if log_writer is not None:
            # Use epoch_1000x as the x-axis in TensorBoard to calibrate curves.
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if data_iter_step % args.log_freq == 0:
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                for key, value in metrics_reduce.items():
                    log_writer.add_scalar('train_' + key, value, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)


def evaluate(model_without_ddp, args, epoch, batch_size=64, log_writer=None):
    was_training = model_without_ddp.training
    model_without_ddp.eval()
    try:
        with use_ema(model_without_ddp):
            _evaluate_ema(model_without_ddp, args, epoch, batch_size, log_writer)
    finally:
        model_without_ddp.train(was_training)


@torch.no_grad()
def _evaluate_ema(model_without_ddp, args, epoch, batch_size, log_writer):

    model_without_ddp.eval()
    device = next(model_without_ddp.parameters()).device
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    num_steps = math.ceil(args.num_images / (batch_size * world_size))

    # Construct the folder name for saving generated images.
    save_folder = os.path.join(
        args.output_dir,
        "epoch{}-dino{}-steps{}-cfg{}-interval{}-{}_{}-steps{}-cfg{}-interval{}-{}-image{}-res{}".format(
            epoch, model_without_ddp.dino_method, model_without_ddp.dino_steps,
            model_without_ddp.dino_cfg_scale, *model_without_ddp.dino_cfg_interval,
            model_without_ddp.method, model_without_ddp.steps, model_without_ddp.cfg_scale,
            model_without_ddp.cfg_interval[0], model_without_ddp.cfg_interval[1], args.num_images, args.img_size
        )
    )
    print("Save to:", save_folder)
    if misc.get_rank() == 0 and not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # ensure that the number of images per class is equal.
    class_num = args.class_num
    assert args.num_images % class_num == 0, "Number of images per class must be the same"
    class_label_gen_world = np.arange(0, class_num).repeat(args.num_images // class_num)

    # join-dino：grounding 由 dino DiT 内部从噪声采样得到（每个样本独立，
    # instance-level），不读验证集，因此 generate 只需要 labels。

    for i in range(num_steps):
        print("Generation step {}/{}".format(i, num_steps))

        gen_indices = np.arange(
            world_size * batch_size * i + local_rank * batch_size,
            world_size * batch_size * i + local_rank * batch_size + batch_size,
        )
        labels_gen = class_label_gen_world[np.minimum(gen_indices, args.num_images - 1)]
        labels_gen = torch.as_tensor(labels_gen, dtype=torch.long, device=device)

        with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            sampled_images = model_without_ddp.generate(labels_gen)

        if misc.is_dist_avail_and_initialized():
            torch.distributed.barrier()

        # denormalize images
        sampled_images = (sampled_images + 1) / 2
        sampled_images = sampled_images.detach().cpu()

        # distributed save images
        for b_id in range(sampled_images.size(0)):
            img_id = i * sampled_images.size(0) * world_size + local_rank * sampled_images.size(0) + b_id
            if img_id >= args.num_images:
                break
            gen_img = np.round(np.clip(sampled_images[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255))
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(os.path.join(save_folder, '{}.png'.format(str(img_id).zfill(5))), gen_img)

    if misc.is_dist_avail_and_initialized():
        torch.distributed.barrier()

    # compute FID and IS
    if log_writer is not None:
        # 统计文件从共享 fid_stats 目录解析，不依赖启动时的当前目录。
        fid_statistics_file = str(fid_stats_path(args.img_size))
        _cache_local_inception_weights(fid_statistics_file)
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=save_folder,
            input2=None,
            fid_statistics_file=fid_statistics_file,
            cuda=device.type == 'cuda',
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=False,
        )
        fid = metrics_dict['frechet_inception_distance']
        inception_score = metrics_dict['inception_score_mean']
        postfix = "_cfg{}_res{}".format(model_without_ddp.cfg_scale, args.img_size)
        log_writer.add_scalar('fid{}'.format(postfix), fid, epoch)
        log_writer.add_scalar('is{}'.format(postfix), inception_score, epoch)
        print("FID: {:.4f}, Inception Score: {:.4f}".format(fid, inception_score))
        shutil.rmtree(save_folder)

    if misc.is_dist_avail_and_initialized():
        torch.distributed.barrier()
