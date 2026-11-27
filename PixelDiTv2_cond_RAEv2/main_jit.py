"""Train or sample ImageNet-256 PixelDiT2-H/16 with frozen RAEv2 grounding."""
import argparse
from pathlib import Path

from resource_paths import PROJECT_ROOT, asset_defaults


def get_args_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.set_defaults(training_stage='PixelDiTv2_cond_RAEv2')
    for key, default in asset_defaults().items():
        p.add_argument('--' + key, default=default)
    p.add_argument('--model', choices=['PixelDiT2-H/16'], default='PixelDiT2-H/16')
    p.add_argument('--condition_mode_pixel', choices=['clean', 'noised', 'pred'], default='clean')
    p.add_argument('--noise_beta_dino_pixel', default=0.25, type=float)
    p.add_argument('--img_size', choices=[256], default=256, type=int)
    p.add_argument('--class_num', choices=[1000], default=1000, type=int)
    p.add_argument('--attn_dropout', default=0., type=float)
    p.add_argument('--proj_dropout', default=0.2, type=float)
    p.add_argument('--label_drop_prob', default=0.1, type=float)
    p.add_argument('--repa_weight', default=0.5, type=float)
    p.add_argument('--repa_align_layer', default=8, type=int)
    p.add_argument('--t_mean_pixel', default=-0.8, type=float)
    p.add_argument('--t_std_pixel', default=0.8, type=float)
    p.add_argument('--t_eps', default=0.05, type=float)
    p.add_argument('--noise_scale_pixel', default=1.0, type=float)
    p.add_argument('--steps_pixel', default=50, type=int)
    p.add_argument('--sampling_method_pixel', choices=['heun'], default='heun')
    p.add_argument('--cfg_pixel', default=2.4, type=float)
    p.add_argument('--interval_min_pixel', default=0.125, type=float)
    p.add_argument('--interval_max_pixel', default=0.9, type=float)
    p.add_argument('--epochs', default=160, type=int)
    p.add_argument('--batch_size', default=128, type=int, help='Training batch size per GPU')
    p.add_argument('--accum_steps', default=1, type=int, help='Batches per optimizer update')
    p.add_argument('--lr', default=2e-4, type=float)
    p.add_argument('--weight_decay', default=0., type=float)
    p.add_argument('--warmup_steps', default=6255, type=int, help='Number of optimizer updates')
    p.add_argument('--ema_decay', default=0.9999, type=float)
    p.add_argument('--num_workers', default=4, type=int)
    p.add_argument('--pin_mem', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--seed', default=0, type=int)
    p.add_argument('--start_epoch', default=0, type=int, help=argparse.SUPPRESS)
    p.add_argument('--output_dir', default='')
    p.add_argument('--resume', default='', help='Training checkpoint or directory containing checkpoint-last.pth')
    p.add_argument('--save_last_freq', default=5, type=int)
    p.add_argument('--save_interval', default=80, type=int)
    p.add_argument('--log_freq', default=50, type=int)
    p.add_argument('--device', default='cuda')
    p.add_argument('--world_size', default=1, type=int)
    p.add_argument('--local_rank', '--local-rank', default=-1, type=int)
    p.add_argument('--dist_on_itp', action='store_true')
    p.add_argument('--dist_url', default='env://')
    p.add_argument('--dist_timeout', default=60, type=int, help='Process group timeout in minutes')
    p.add_argument('--online_eval', action='store_true')
    p.add_argument('--eval_freq', default=20, type=int)
    p.add_argument('--evaluate_gen', action='store_true')
    p.add_argument('--checkpoint', default='', help='Pixel checkpoint for generation')
    p.add_argument('--samples_only', action='store_true', help='Keep generated PNGs and skip FID/IS')
    p.add_argument('--num_images', default=50000, type=int)
    p.add_argument('--gen_bsz', default=64, type=int)
    return p


def validate_args(args):
    from util.training import resolve_checkpoint
    for name in ('epochs', 'batch_size', 'accum_steps', 'log_freq', 'save_interval', 'save_last_freq',
                 'num_images', 'gen_bsz', 'steps_pixel', 'eval_freq', 'dist_timeout', 'lr', 'noise_scale_pixel'):
        if getattr(args, name) <= 0:
            raise ValueError(f'{name} must be positive')
    if min(args.num_workers, args.warmup_steps, args.repa_weight, args.t_std_pixel, args.weight_decay) < 0:
        raise ValueError('Worker count, warmup, REPA weight, time std and weight decay must be nonnegative')
    if not 0 <= args.ema_decay < 1 or not 0 < args.t_eps <= 1:
        raise ValueError('Require 0 <= ema_decay < 1 and 0 < t_eps <= 1')
    if not 0 < args.noise_beta_dino_pixel <= 1:
        raise ValueError('noise_beta_dino_pixel must be in (0, 1]')
    if not 0 <= args.interval_min_pixel <= args.interval_max_pixel <= 1:
        raise ValueError('Invalid Pixel CFG interval')
    if not 1 <= args.repa_align_layer <= 32:
        raise ValueError('REPA layer must be between 1 and 32')
    if any(not 0 <= prob <= 1 for prob in (args.attn_dropout, args.proj_dropout, args.label_drop_prob)):
        raise ValueError('Dropout probabilities must be within [0, 1]')
    if args.checkpoint and (args.resume or not args.evaluate_gen):
        raise ValueError('Use --resume for training or --evaluate_gen --checkpoint for sampling')
    checkpoint = resolve_checkpoint(args.checkpoint or args.resume)
    if args.evaluate_gen and checkpoint is None:
        raise ValueError('--evaluate_gen needs a Pixel checkpoint')
    required = [args.dino_stat_path]
    if not args.evaluate_gen:
        required.extend([Path(args.data_path) / 'train', Path(args.dino_encoder_repo) / 'hubconf.py',
                         args.dino_encoder_weights_path])
        if args.repa_weight:
            required.append(args.repa_encoder_weights_path)
    if args.online_eval or args.evaluate_gen or args.condition_mode_pixel == 'pred':
        required.extend([args.raev2_checkpoint, args.raev2_config])
        if not args.samples_only:
            required.append(args.fid_stats)
            if args.num_images % args.class_num:
                raise ValueError('Class-balanced FID/IS needs a multiple of 1000 images')
    for path in required:
        if not path or not Path(path).expanduser().exists():
            raise FileNotFoundError(f'Required resource missing: {path}')
    return checkpoint


def main(args):
    import torch
    from checkpoint_io import initialize_ema, latent_spec, read_checkpoint, resume_model
    from denoiser import Denoiser
    from engine_jit import evaluate
    from util.train_epoch import train_one_epoch
    from util.training import initialize_run, make_log_writer, make_optimizer, make_train_loader, save_epoch, wrap_distributed

    checkpoint = validate_args(args)
    if not args.output_dir:
        if checkpoint:
            out = checkpoint.parent / f'eval-{checkpoint.stem}' if args.evaluate_gen else checkpoint.parent
        else:
            condition_suffix = '' if args.condition_mode_pixel == 'clean' else f'-{args.condition_mode_pixel}'
            out = PROJECT_ROOT / 'checkpoints' / f'raev2-dinov3l-k7-pixeldit2-h16{condition_suffix}-e{args.epochs}'
        args.output_dir = str(out)
    args.latent_spec = latent_spec(args)
    device = initialize_run(args)
    try:
        model = Denoiser(args).to(device)
        wrapper = model if args.evaluate_gen else wrap_distributed(model, args)
        optimizer = None if args.evaluate_gen else make_optimizer(model, args)
        if checkpoint:
            resume_model(model, optimizer, read_checkpoint(checkpoint), args)
        else:
            initialize_ema(model)
        writer = make_log_writer(args.output_dir)
        try:
            if args.evaluate_gen:
                evaluate(model, args, args.start_epoch, args.gen_bsz, writer)
                return
            loader = make_train_loader(args)
            for epoch in range(args.start_epoch, args.epochs):
                loader.sampler.set_epoch(epoch)
                train_one_epoch(wrapper, model, loader, optimizer, device, epoch, writer, args)
                completed = epoch + 1
                save_epoch(args, model, optimizer, completed)
                if args.online_eval and (completed % args.eval_freq == 0 or completed == args.epochs):
                    evaluate(model, args, completed, args.gen_bsz, writer)
                if writer:
                    writer.flush()
        finally:
            if writer:
                writer.close()
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main(get_args_parser().parse_args())
