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
import copy
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

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = model(x, labels)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        model_without_ddp.update_ema()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None:
            # Use epoch_1000x as the x-axis in TensorBoard to calibrate curves.
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if data_iter_step % args.log_freq == 0:
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)


def evaluate(model_without_ddp, args, epoch, batch_size=64, log_writer=None):

    model_without_ddp.eval()
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    num_steps = args.num_images // (batch_size * world_size) + 1

    # Construct the folder name for saving generated images.
    save_folder = os.path.join(
        args.output_dir,
        "{}-steps{}-cfg{}-interval{}-{}-image{}-res{}".format(
            model_without_ddp.method, model_without_ddp.steps, model_without_ddp.cfg_scale,
            model_without_ddp.cfg_interval[0], model_without_ddp.cfg_interval[1], args.num_images, args.img_size
        )
    )
    print("Save to:", save_folder)
    if misc.get_rank() == 0 and not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # 切换到 EMA1 采样；冻结的 DINOv3 不需要单独的 EMA 副本。
    model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    trainable_parameters = [(name, value) for name, value in model_without_ddp.named_parameters()
                            if value.requires_grad]
    for i, (name, _value) in enumerate(trainable_parameters):
        assert name in ema_state_dict
        ema_state_dict[name] = model_without_ddp.ema_params1[i]
    print("Switch to ema")
    model_without_ddp.load_state_dict(ema_state_dict)

    # ensure that the number of images per class is equal.
    class_num = args.class_num
    assert args.num_images % class_num == 0, "Number of images per class must be the same"
    class_label_gen_world = np.arange(0, class_num).repeat(args.num_images // class_num)
    class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(50000)])

    # 与 class_label_gen_world 一一对应的类内序号：全局第 g 张生成图属于
    # 类别 g // per_class，取该类别验证集的第 g % per_class 张真实图片作为
    # 语义来源。50000 / 1000 = 50，正好覆盖每类 50 张验证图。
    per_class = args.num_images // class_num
    use_val_grounding = (
        getattr(model_without_ddp, "grounding_source", "none") == "val"
        and getattr(model_without_ddp, "use_grounding", False)
    )
    # class-level：每个类别用固定种子随机抽一张验证图，该类全部样本共用这一份
    # grounding（c 相同、噪声不同）。种子与 rank 无关，所有进程得到同一张表。
    val_grounding_level = getattr(args, "val_grounding_level", "instance")
    if use_val_grounding and val_grounding_level == "class":
        class_ref_index = np.random.default_rng(args.class_ref_seed).integers(0, per_class, size=class_num)
        print("Class-level grounding: seed {}, class 0-4 use val index {}".format(
            args.class_ref_seed, class_ref_index[:5].tolist()))

    for i in range(num_steps):
        print("Generation step {}/{}".format(i, num_steps))

        gen_indices = np.arange(
            world_size * batch_size * i + local_rank * batch_size,
            world_size * batch_size * i + local_rank * batch_size + batch_size,
        )
        labels_gen = class_label_gen_world[gen_indices]
        labels_gen = torch.Tensor(labels_gen).long().cuda()
        # 类内序号：同一类别内生成第几张，用来配对验证集真实图片。
        intra_indices = gen_indices % per_class
        if use_val_grounding and val_grounding_level == "class":
            # 补齐用的尾部（gen_indices >= num_images，标签为 0、不保存）同样按类别取。
            intra_indices = class_ref_index[class_label_gen_world[gen_indices].astype(np.int64)]

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            gen_kwargs = {}
            if use_val_grounding:
                gen_kwargs = dict(intra_indices=intra_indices, data_path=args.data_path)
            sampled_images = model_without_ddp.generate(labels_gen, **gen_kwargs)

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

    torch.distributed.barrier()

    # back to no ema
    print("Switch back from ema")
    model_without_ddp.load_state_dict(model_state_dict)

    # compute FID and IS
    if log_writer is not None:
        # 统计文件从共享 fid_stats 目录解析，不依赖启动时的当前目录。
        fid_statistics_file = str(fid_stats_path(args.img_size))
        _cache_local_inception_weights(fid_statistics_file)
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=save_folder,
            input2=None,
            fid_statistics_file=fid_statistics_file,
            cuda=True,
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

    torch.distributed.barrier()
