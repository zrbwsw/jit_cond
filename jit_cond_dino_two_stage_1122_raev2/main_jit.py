"""Stage 2 的 Pixel JiT 训练与采样（使用 RAEv2 DINOv3-L k=7 encoder）。"""

import argparse
from pathlib import Path
import time

import torch

from checkpoint_io import initialize_ema, latent_spec, load_branch, read_checkpoint, resume_model
from denoiser import Denoiser
from engine_jit import evaluate
from util.train_epoch import train_one_epoch
from util.training import (
    add_training_args, initialize_run, make_log_writer, make_optimizer, make_train_loader,
    resolve_checkpoint, save_epoch, validate_training_args, wrap_distributed,
)


def get_args_parser():
    parser = argparse.ArgumentParser(description='Stage 2: train Pixel JiT with frozen DINOv3-L k=7 encoder')
    parser.set_defaults(training_stage='pixel')
    add_training_args(parser, epochs=200)

    # DINOv3-L k=7 encoder 参数
    parser.add_argument('--dino_stat_path', required=True,
                        help='Path to dinov3l_k7_imagenet1k_stats.pt (from RAEv2)')
    parser.add_argument('--dino_encoder_weights_path', default=None,
                        help='Path to dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth')
    parser.add_argument('--dino_encoder_repo', default=None,
                        help='Path to DINOv3 source directory (containing hubconf.py)')
    parser.add_argument('--raev2_checkpoint', default='',
                        help='RAEv2 stage-2 EMA checkpoint used for latent sampling')
    parser.add_argument('--raev2_config', default='',
                        help='RAEv2 sampling YAML used for latent sampling')

    # Pixel JiT 参数
    parser.add_argument('--model', default='JiT-B/16')
    parser.add_argument('--t_mean_pixel', '--P_mean', default=-0.8, type=float)
    parser.add_argument('--t_std_pixel', '--P_std', default=0.8, type=float)
    parser.add_argument('--noise_scale_pixel', default=1.0, type=float)
    parser.add_argument('--steps_pixel', '--num_sampling_steps', default=50, type=int)
    parser.add_argument('--sampling_method_pixel', '--sampling_method', default='heun', choices=['euler', 'heun'])
    parser.add_argument('--cfg_pixel', '--cfg', default=2.9, type=float)
    parser.add_argument('--interval_min_pixel', '--interval_min', default=0.1, type=float)
    parser.add_argument('--interval_max_pixel', '--interval_max', default=1.0, type=float)

    # 评估参数
    parser.add_argument('--dist_timeout', default=60, type=int)
    parser.add_argument('--num_images', default=50000, type=int)
    parser.add_argument('--gen_bsz', default=64, type=int)
    parser.add_argument('--eval_freq', default=20, type=int)
    parser.add_argument('--online_eval', action='store_true')
    parser.add_argument('--evaluate_gen', action='store_true')
    parser.add_argument('--checkpoint', default='', help='Exact Stage 2 checkpoint for --evaluate_gen')

    # 初始化参数（可选）
    parser.add_argument('--init_pixel', default='', help='Optional Pixel weight initialization')
    parser.add_argument('--init_pixel_weights', choices=['model', 'ema1', 'ema2'], default='model')

    return parser


def validate_args(args):
    validate_training_args(args)

    # 检查统计文件
    if not Path(args.dino_stat_path).is_file():
        raise FileNotFoundError(
            f"DINOv3-L k=7 stats not found: {args.dino_stat_path}. "
            "Download from https://huggingface.co/nyu-visionx/RAEv2-models/blob/main/stage1/imagenet/dinov3l-k7/stats.pt"
        )

    if min(args.noise_scale_pixel, args.steps_pixel, args.eval_freq, args.gen_bsz, args.num_images) <= 0:
        raise ValueError('Pixel noise scale, sampling steps, evaluation frequency and sample sizes must be positive')

    if args.checkpoint and not args.evaluate_gen:
        raise ValueError('--checkpoint is for sampling; use --resume to continue training')

    checkpoint_path = resolve_checkpoint(args.checkpoint or args.resume)
    if checkpoint_path and args.init_pixel:
        raise ValueError('Resume uses weights from checkpoint; cannot also set --init_pixel')
    elif args.evaluate_gen and not checkpoint_path:
        raise ValueError('--evaluate_gen requires --checkpoint or --resume')
    if (args.online_eval or args.evaluate_gen) and (not args.raev2_checkpoint or not args.raev2_config):
        raise ValueError('RAEv2 sampling requires --raev2_checkpoint and --raev2_config')

    return checkpoint_path


def main(args):
    checkpoint_path = validate_args(args)

    # 设置输出目录
    if not args.output_dir:
        if checkpoint_path:
            folder = checkpoint_path.parent
            if args.evaluate_gen:
                folder /= f'eval-{checkpoint_path.stem}'
        else:
            # 新训练：使用 RAEv2 dinov3l-k7
            pixel_size = args.model.removeprefix('JiT-').split('/', 1)[0].lower()
            name = f'stage1-raev2-dinov3l-k7-stage2-pixeljit-{pixel_size}-e{args.epochs}'
            folder = Path('checkpoints') / name
        args.output_dir = str(folder)

    args.latent_spec = latent_spec(args)
    device, seed = initialize_run(args)

    # 创建模型（DINOv3-L k=7 encoder 在 Denoiser.__init__ 中加载）
    model = Denoiser(args)

    # 可选：从其他 Pixel JiT checkpoint 初始化
    if args.init_pixel:
        load_branch(model, read_checkpoint(args.init_pixel), 'net', args.init_pixel_weights, args)

    model.to(device)

    # 分布式训练包装
    distributed_model = model if args.evaluate_gen else wrap_distributed(model, args)
    optimizer = None if args.evaluate_gen else make_optimizer(model, args)

    # Resume 或初始化 EMA
    if checkpoint_path:
        resume_model(model, optimizer, read_checkpoint(checkpoint_path), args)
        print(f'Loaded Pixel checkpoint: {checkpoint_path}')
    else:
        initialize_ema(model)

    log_writer = make_log_writer(args.output_dir)
    try:
        if args.evaluate_gen:
            devices = [torch.cuda.current_device()] if device.type == 'cuda' else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(seed)
                evaluate(model, args, args.start_epoch, batch_size=args.gen_bsz, log_writer=log_writer)
            return

        loader = make_train_loader(args)
        start_time = time.monotonic()
        print(f'Start Pixel JiT training for {args.epochs} epochs')
        for epoch in range(args.start_epoch, args.epochs):
            completed = epoch + 1
            loader.sampler.set_epoch(epoch)
            train_one_epoch(distributed_model, model, loader, optimizer, device, epoch, log_writer, args)
            save_epoch(args, model, optimizer, completed)
            if args.online_eval and (completed % args.eval_freq == 0 or completed == args.epochs):
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                evaluate(model, args, completed, batch_size=args.gen_bsz, log_writer=log_writer)
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            if log_writer is not None:
                log_writer.flush()
        print(f'Pixel JiT training time: {time.monotonic() - start_time:.1f}s')
    finally:
        if log_writer is not None:
            log_writer.close()


if __name__ == '__main__':
    main(get_args_parser().parse_args())
