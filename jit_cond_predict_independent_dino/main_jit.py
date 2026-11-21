import argparse
import datetime
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from util.crop import center_crop_arr
import util.misc as misc

from engine_jit import train_one_epoch, evaluate

from denoiser import Denoiser
from denoiser_dino import DinoDenoiser
from checkpoint_io import initialize_ema, latent_spec, load_branch, read_checkpoint, resume_model
from resource_paths import imagenet_root


def get_args_parser():
    parser = argparse.ArgumentParser('Independent DINO / predicted-condition Pixel JiT')
    parser.set_defaults(training_stage='pixel')

    # 模型结构；use_grounding 打开后，Denoiser 会创建冻结的 DINOv3 teacher
    # 和 JiT 的 patch-level AdaLN 投影层。
    parser.add_argument('--model', default='JiT-B/16', type=str, metavar='MODEL',
                        help='Name of the model to train')
    parser.add_argument('--img_size', default=256, type=int, help='Image size')
    parser.add_argument('--attn_dropout', type=float, default=0.0, help='Attention dropout rate')
    parser.add_argument('--proj_dropout', type=float, default=0.0, help='Projection dropout rate')
    parser.add_argument('--use_grounding', '--enable_grounding', action=argparse.BooleanOptionalAction, default=True,
                        help='Use DINO conditioning (required in this experiment)')
    parser.add_argument('--grounding_dim', default=768, type=int,
                        help='grounding/dino latent 的通道数（DINOv2-reg-B = 768）')
    # dino path（独立 DiT 去噪 DINOv2 latent，结构/编码与 Latent Forcing 对齐）
    parser.add_argument('--dino_model', default='DinoDiT-B', type=str,
                        help='dino DiT 规模；DinoDiT-B 与 JiT-B 同尺寸（12 层/768/12 头）')
    parser.add_argument('--dino_encoder', default='dinov2b', choices=['dinov2b', 'dinov3b'], type=str,
                        help='冻结的 latent 编码器 / dino DiT 目标空间：dinov2b = DINOv2-reg-B/14，'
                             'dinov3b = DINOv3-B/16；两者最后 norm 均去 affine，归一化不乘 0.485')
    parser.add_argument('--dinov3_weights', default='', type=str,
                        help='dinov3b 时的本地 DINOv3 权重路径；留空按 PRETRAIN_MODELS_ROOT 解析')
    parser.add_argument('--dinov3_repo', default='', type=str,
                        help='dinov3b 时的本地 DINOv3 源码目录（含 hubconf.py）；留空按 DINOV3_REPO 解析')
    parser.add_argument('--dino_stat_path', default='', type=str,
                        help='latent 统计文件；留空按 --dino_encoder 用 stat/ 下对应文件')
    parser.add_argument('--D_mean', default=-1.2, type=float, help='dino path t 采样的 logit-normal 均值（LF 默认）')
    parser.add_argument('--D_std', default=1.0, type=float, help='dino path t 采样的 logit-normal 标准差（LF 默认）')
    parser.add_argument('--dino_loss_scale', default=1.0, type=float,
                        help='Standalone DINO loss scale; 0.333 reproduces the old loss multiplier')
    parser.add_argument('--dino_noise_beta', default=0.25, type=float,
                        help='Noise amplitude before ONE frozen DinoDiT prediction; s in [1-beta,1)')
    parser.add_argument('--condition_time_schedule', choices=['uniform', 'logit_normal'], default='uniform',
                        help='Pixel condition input s; logit_normal uses D_mean/D_std instead of beta')
    parser.add_argument('--dino_checkpoint', default='',
                        help='Trusted joint/DINO checkpoint; REQUIRED for a fresh Pixel run')
    parser.add_argument('--dino_weights', choices=['model', 'ema1', 'ema2'], default='ema1',
                        help='DINO branch weights to extract (EMA1 by default)')
    parser.add_argument('--init_pixel', default='', help='Optional Pixel branch warm start, NOT a resume')
    parser.add_argument('--init_pixel_weights', choices=['model', 'ema1', 'ema2'], default='model')
    parser.add_argument('--dino_steps', default=25, type=int, help='采样时 dino DiT 的积分步数')
    parser.add_argument('--dino_sampling_method', default='heun', choices=['euler', 'heun'], type=str)
    parser.add_argument('--dino_cfg', default=1.0, type=float, help='dino path 采样 CFG；1.0=不加引导')
    parser.add_argument('--dino_interval_min', default=0.0, type=float)
    parser.add_argument('--dino_interval_max', default=1.0, type=float)

    # training
    parser.add_argument('--epochs', default=80, type=int)
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='Epochs to warm up LR')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size = batch_size * # GPUs)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='Learning rate (absolute)')
    parser.add_argument('--blr', type=float, default=5e-5, metavar='LR',
                        help='Base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='Minimum LR for cyclic schedulers that hit 0')
    parser.add_argument('--lr_schedule', type=str, default='constant',
                        help='Learning rate schedule')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay (default: 0.0)')
    parser.add_argument('--ema_decay1', type=float, default=0.9999,
                        help='The first ema to track. Use the first ema for sampling by default.')
    parser.add_argument('--ema_decay2', type=float, default=0.9996,
                        help='The second ema to track')
    parser.add_argument('--P_mean', default=-0.8, type=float)
    parser.add_argument('--P_std', default=0.8, type=float)
    parser.add_argument('--noise_scale', default=1.0, type=float)
    parser.add_argument('--t_eps', default=5e-2, type=float)
    parser.add_argument('--label_drop_prob', default=0.1, type=float)

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='Starting epoch')
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for faster GPU transfers')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # sampling
    parser.add_argument('--sampling_method', default='heun', type=str,
                        help='ODE samping method')
    parser.add_argument('--num_sampling_steps', default=50, type=int,
                        help='Sampling steps')
    parser.add_argument('--cfg', default=1.0, type=float,
                        help='Classifier-free guidance factor')
    parser.add_argument('--interval_min', default=0.0, type=float,
                        help='CFG interval min')
    parser.add_argument('--interval_max', default=1.0, type=float,
                        help='CFG interval max')
    parser.add_argument('--num_images', default=50000, type=int,
                        help='Number of images to generate')
    parser.add_argument('--eval_freq', type=int, default=40,
                        help='Frequency (in epochs) for evaluation')
    parser.add_argument('--online_eval', action='store_true')
    parser.add_argument('--evaluate_gen', action='store_true')
    parser.add_argument('--gen_bsz', type=int, default=256,
                        help='Generation batch size')

    # 数据路径默认指向共享 ImageNet；训练仍使用原始 ImageFolder 结构。
    parser.add_argument('--data_path', default=str(imagenet_root()), type=str,
                        help='Path to the dataset')
    parser.add_argument('--class_num', default=1000, type=int)

    # checkpoint：--checkpoint 用于采样指定某个文件，--resume 默认查找
    # checkpoint-last.pth；两者都兼容 JiT-cond 自己保存的 EMA 结构。
    parser.add_argument('--output_dir', default='./checkpoints/pixel_predict',
                        help='Directory to save outputs (empty for no saving)')
    parser.add_argument('--resume', default='',
                        help='Same-stage independent checkpoint file or folder; must exist')
    parser.add_argument('--checkpoint', default='',
                        help='Exact checkpoint file for --evaluate_gen (overrides --resume/checkpoint-last.pth)')
    parser.add_argument('--save_last_freq', type=int, default=5,
                        help='Frequency (in epochs) to save checkpoints')
    parser.add_argument('--save_interval', type=int, default=20,
                        help='Frequency (in epochs) to save checkpoints')
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training/testing')

    # distributed training
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='URL used to set up distributed training')

    return parser


def validate_args(args):
    if not args.use_grounding:
        raise ValueError('This experiment requires DINO conditioning')
    if args.img_size != 256 or args.grounding_dim != 768:
        raise ValueError('The supplied DINO encoders/statistics use 256 images and 768x16x16 latents')
    if not 0 < args.dino_noise_beta <= 1:
        raise ValueError('dino_noise_beta must be in (0,1]')
    if args.t_eps <= 0 or args.noise_scale <= 0 or args.dino_loss_scale <= 0:
        raise ValueError('t_eps, noise_scale and dino_loss_scale must be positive')
    if min(args.dino_steps, args.num_sampling_steps, args.save_last_freq, args.save_interval, args.eval_freq) < 1:
        raise ValueError('Sampling steps and save/eval frequencies must be positive')
    if args.training_stage == 'dino' and (args.online_eval or args.evaluate_gen or args.init_pixel):
        raise ValueError('DINO-only training has no Pixel image evaluation or Pixel initialization')
    if args.checkpoint and not args.evaluate_gen:
        raise ValueError('--checkpoint is for --evaluate_gen; use --resume to resume training')
    path = args.checkpoint or args.resume
    if path:
        path = Path(path).expanduser()
        if path.is_dir():
            path /= 'checkpoint-last.pth'
        if not path.is_file():
            raise FileNotFoundError(f'Resume/evaluation checkpoint not found: {path}')
        if args.dino_checkpoint or args.init_pixel:
            raise ValueError('Resume/evaluation restores the full pipeline; do not also supply initialization weights')
    elif args.evaluate_gen:
        raise ValueError('--evaluate_gen requires --checkpoint or --resume')
    elif args.training_stage == 'pixel' and not args.dino_checkpoint:
        raise ValueError('A fresh Pixel run requires --dino_checkpoint; random frozen DINO is not allowed')
    for initial in (args.dino_checkpoint, args.init_pixel):
        if initial and not Path(initial).expanduser().is_file():
            raise FileNotFoundError(f'Initialization checkpoint not found: {initial}')
    return path or None


def main(args):
    checkpoint_path = validate_args(args)
    args.latent_spec = latent_spec(args)
    misc.init_distributed_mode(args)
    print('Job directory:', os.path.dirname(os.path.realpath(__file__)))
    print("Arguments:\n{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # Set seeds for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    # Set up TensorBoard logging (only on main process)
    if global_rank == 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    # Data augmentation transforms
    transform_train = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.PILToTensor()
    ])

    if not args.evaluate_gen:
        dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
        print(dataset_train)
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train, batch_size=args.batch_size,
            num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True)

    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.optimize_ddp = False

    # Create denoiser
    model = DinoDenoiser(args) if args.training_stage == 'dino' else Denoiser(args)
    if args.dino_checkpoint:
        load_branch(model, read_checkpoint(args.dino_checkpoint), 'dino_net', args.dino_weights, args)
    if args.init_pixel:
        load_branch(model, read_checkpoint(args.init_pixel), 'net', args.init_pixel_weights, args)

    print("Model =", model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {:.6f}M".format(n_params / 1e6))

    model.to(device)

    eff_batch_size = args.batch_size * misc.get_world_size()
    if args.lr is None:  # only base_lr (blr) is specified
        args.lr = args.blr * eff_batch_size / 256

    print("Base lr: {:.2e}".format(args.lr * 256 / eff_batch_size))
    print("Actual lr: {:.2e}".format(args.lr))
    print("Effective batch size: %d" % eff_batch_size)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # Set up optimizer with weight decay adjustment for bias and norm layers
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)

    if checkpoint_path:
        resume_model(model_without_ddp, optimizer, read_checkpoint(checkpoint_path), args)
        print('Resumed', args.training_stage, 'checkpoint:', checkpoint_path)
    else:
        initialize_ema(model_without_ddp)
        print('Fresh optimizer/EMA for', args.training_stage, '(initial weights as specified above)')

    # Evaluate generation
    if args.evaluate_gen:
        print("Evaluating checkpoint at {} epoch".format(args.start_epoch))
        with torch.random.fork_rng(devices=[device.index or torch.cuda.current_device()] if device.type == 'cuda' else []):
            torch.manual_seed(seed)
            with torch.no_grad():
                evaluate(model_without_ddp, args, args.start_epoch, batch_size=args.gen_bsz, log_writer=log_writer)
        if log_writer is not None:
            log_writer.close()
        return

    # Training loop
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(model, model_without_ddp, data_loader_train, optimizer, device, epoch, log_writer=log_writer, args=args)

        # Save final checkpoint periodically for easier resuming
        if (epoch + 1) % args.save_last_freq == 0 or (epoch + 1) == args.epochs:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=(epoch + 1),
                epoch_name="last"
            )

        # Every save_interval epochs, we save a persistent checkpoint
        if (epoch + 1) % args.save_interval == 0:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=(epoch + 1)
            )

        # Perform online evaluation at specified intervals
        if args.online_eval and ((epoch + 1) % args.eval_freq == 0 or epoch + 1 == args.epochs):
            torch.cuda.empty_cache()
            with torch.no_grad():
                evaluate(model_without_ddp, args, epoch + 1, batch_size=args.gen_bsz, log_writer=log_writer)
            torch.cuda.empty_cache()

        if misc.is_main_process() and log_writer is not None:
            log_writer.flush()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time:', total_time_str)
    if log_writer is not None:
        log_writer.close()


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
