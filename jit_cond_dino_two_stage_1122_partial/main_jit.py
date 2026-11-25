"""Stage 2 的 Pixel JiT 训练与两阶段图像采样。"""

import argparse
from pathlib import Path
import time

import torch

from checkpoint_io import completed_epochs, initialize_ema, latent_spec, load_branch, read_checkpoint, resume_model
from denoiser import Denoiser
from engine_jit import evaluate
from stage1.config_dino import add_dino_args, experiment_name_dino, validate_dino_args
from util.train_epoch import train_one_epoch
from util.training import (
    add_training_args, initialize_run, make_log_writer, make_optimizer, make_train_loader,
    resolve_checkpoint, save_epoch, validate_training_args, wrap_distributed,
)


def get_args_parser():
    parser = argparse.ArgumentParser(description='Stage 2: train Pixel JiT with a frozen DinoDiT')
    parser.set_defaults(training_stage='pixel')
    add_training_args(parser, epochs=200)
    add_dino_args(parser)
    parser.add_argument('--model', default='JiT-B/16')
    parser.add_argument('--verify_cache_hashes', action='store_true')
    parser.add_argument('--grounding_cache', default='', help='Completed train cache directory')
    parser.add_argument('--cache_cfg_dino', default=None, type=float,
                        help='CFG used to create the offline cache; defaults to cfg_dino for compatibility')
    parser.add_argument('--condition_mode_pixel', choices=['noised', 'pred', 'cached_recon', 'partial_trajectory'], default='pred',
                        help='Stage 2 grounding: pred (single-step, default) | partial_trajectory (t_part→1 sampling) | '
                             'noised (U[0.75,1) noise) | cached_recon (v3)')
    parser.add_argument('--noise_beta_dino_pixel', '--dino_noise_beta', default=0.25, type=float)
    parser.add_argument('--t_schedule_dino_pixel', '--condition_time_schedule',
                        choices=['uniform', 'logit_normal'], default='uniform')
    # P2: logit_normal 分支专用参数，与 Stage 1 的 t_mean_dino/t_std_dino 独立。
    # 默认 mean=1.95 ≈ logit(0.875) 对齐 U[0.75,1) 中值，std=0.5 收紧到 [0.75,1) 附近。
    # uniform 分支忽略这两个值。
    parser.add_argument('--t_mean_dino_pixel', default=1.95, type=float)
    parser.add_argument('--t_std_dino_pixel', default=0.5, type=float)
    # ===== partial trajectory 参数（已废弃单一 t_part，改为训练时随机采样）=====
    parser.add_argument('--t_part', default=0.8, type=float,
                        help='[Deprecated] Mean of t_part range; training uses U[0.6, 1.0]')
    parser.add_argument('--steps_part', default=5, type=int,
                        help='Heun steps from t_part to 1.0 for partial_trajectory (default 5)')
    # =====
    parser.add_argument('--t_mean_pixel', '--P_mean', default=-0.8, type=float)
    parser.add_argument('--t_std_pixel', '--P_std', default=0.8, type=float)
    parser.add_argument('--noise_scale_pixel', default=1.0, type=float)
    parser.add_argument('--dino_checkpoint', default='', help='Required Stage 1 or compatible join-dino checkpoint')
    parser.add_argument('--dino_weights', choices=['model', 'ema1', 'ema2'], default='ema1')
    parser.add_argument('--init_pixel', default='', help='Optional Pixel weight initialization')
    parser.add_argument('--init_pixel_weights', choices=['model', 'ema1', 'ema2'], default='model')
    parser.add_argument('--steps_pixel', '--num_sampling_steps', default=50, type=int)
    parser.add_argument('--sampling_method_pixel', '--sampling_method', default='heun', choices=['euler', 'heun'])
    parser.add_argument('--cfg_pixel', '--cfg', default=2.9, type=float)
    parser.add_argument('--interval_min_pixel', '--interval_min', default=0.1, type=float)
    parser.add_argument('--interval_max_pixel', '--interval_max', default=1.0, type=float)
    parser.add_argument('--num_images', default=50000, type=int)
    parser.add_argument('--gen_bsz', default=64, type=int)
    parser.add_argument('--eval_freq', default=20, type=int)
    parser.add_argument('--online_eval', action='store_true')
    parser.add_argument('--evaluate_gen', action='store_true')
    parser.add_argument('--checkpoint', default='', help='Exact Stage 2 checkpoint for --evaluate_gen')
    return parser


def validate_args(args):
    validate_dino_args(args)
    validate_training_args(args)
    if not 0 < args.noise_beta_dino_pixel <= 1:
        raise ValueError('noise_beta_dino_pixel must be in (0, 1]')
    if min(args.noise_scale_pixel, args.steps_pixel, args.eval_freq, args.gen_bsz, args.num_images) <= 0:
        raise ValueError('Pixel noise scale, sampling steps, evaluation frequency and sample sizes must be positive')
    if args.checkpoint and not args.evaluate_gen:
        raise ValueError('--checkpoint is for sampling; use --resume to continue training')
    checkpoint_path = resolve_checkpoint(args.checkpoint or args.resume)
    if checkpoint_path:
        if args.dino_checkpoint or args.init_pixel:
            raise ValueError('Resume/sampling uses the teacher saved in the Stage 2 checkpoint')
    elif args.evaluate_gen:
        raise ValueError('--evaluate_gen requires --checkpoint or --resume')
    elif not args.dino_checkpoint:
        raise ValueError('A fresh Pixel run requires --dino_checkpoint')
    if not args.evaluate_gen and args.condition_mode_pixel == 'cached_recon' and not args.grounding_cache:
        raise ValueError('cached_recon training requires --grounding_cache')
    return checkpoint_path


def main(args):
    checkpoint_path = validate_args(args)
    initial_dino = read_checkpoint(args.dino_checkpoint) if args.dino_checkpoint else None
    if not args.output_dir:
        if checkpoint_path:
            folder = checkpoint_path.parent
            if args.evaluate_gen:
                folder /= f'eval-{checkpoint_path.stem}'
        else:
            stage1_name = experiment_name_dino(args.dino_encoder, args.dino_model, completed_epochs(initial_dino))
            pixel_size = args.model.removeprefix('JiT-').split('/', 1)[0].lower()
            name = f'{stage1_name}-stage2-pixeljit-{pixel_size}-dinodit-{args.condition_mode_pixel}-e{args.epochs}'
            folder = Path('checkpoints') / name
        args.output_dir = str(folder)
    args.latent_spec = latent_spec(args)
    device, seed = initialize_run(args)

    model = Denoiser(args)
    if initial_dino is not None:
        load_branch(model, initial_dino, 'dino_net', args.dino_weights, args)
        del initial_dino
    if args.init_pixel:
        load_branch(model, read_checkpoint(args.init_pixel), 'net', args.init_pixel_weights, args)
    model.to(device)
    # 先同步各 rank 的 Pixel 初始化，再复制 EMA 初值。
    distributed_model = model if args.evaluate_gen else wrap_distributed(model, args)
    optimizer = None if args.evaluate_gen else make_optimizer(model, args)
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

        if args.condition_mode_pixel == 'cached_recon':
            from util.recon_cache import cached_dataset
            dataset = cached_dataset(args, model)
            loader = make_train_loader(args, dataset=dataset)
        else:
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
