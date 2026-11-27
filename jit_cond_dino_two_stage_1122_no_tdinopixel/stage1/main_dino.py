"""Stage 1 训练入口。运行 python -m stage1.main_dino。"""

import argparse
from pathlib import Path
import time

from checkpoint_io import initialize_ema, latent_spec, read_checkpoint, resume_model
from stage1.config_dino import add_dino_args, experiment_name_dino, validate_dino_args
from stage1.denoiser_dino import DinoDenoiser
from util.train_epoch import train_one_epoch
from util.training import (
    add_training_args, initialize_run, make_log_writer, make_optimizer, make_train_loader,
    resolve_checkpoint, save_epoch, validate_training_args, wrap_distributed,
)


def get_args_parser():
    parser = argparse.ArgumentParser(description='Stage 1: train DinoDiT in normalized DINO latent space')
    parser.set_defaults(training_stage='dino')
    add_training_args(parser, epochs=80)
    add_dino_args(parser)
    parser.add_argument('--loss_scale_dino', '--dino_loss_scale', default=1.0, type=float)
    return parser


def validate_args(args):
    validate_dino_args(args)
    validate_training_args(args)
    if args.loss_scale_dino <= 0:
        raise ValueError('loss_scale_dino must be positive')
    return resolve_checkpoint(args.resume)


def main(args):
    resume_path = validate_args(args)
    if not args.output_dir:
        name = experiment_name_dino(args.dino_encoder, args.dino_model, args.epochs)
        args.output_dir = str(resume_path.parent if resume_path else Path('checkpoints') / name)
    args.latent_spec = latent_spec(args)
    device, _ = initialize_run(args)
    loader = make_train_loader(args)
    model = DinoDenoiser(args).to(device)
    # 先同步各 rank 的模型初始化，再复制 EMA 初值。
    distributed_model = wrap_distributed(model, args)
    optimizer = make_optimizer(model, args)
    if resume_path:
        resume_model(model, optimizer, read_checkpoint(resume_path), args)
        print(f'Resumed DinoDiT checkpoint: {resume_path}')
    else:
        initialize_ema(model)
    log_writer = make_log_writer(args.output_dir)

    start_time = time.monotonic()
    print(f'Start DinoDiT training for {args.epochs} epochs')
    try:
        for epoch in range(args.start_epoch, args.epochs):
            loader.sampler.set_epoch(epoch)
            train_one_epoch(distributed_model, model, loader, optimizer, device, epoch, log_writer, args)
            save_epoch(args, model, optimizer, epoch + 1)
            if log_writer is not None:
                log_writer.flush()
    finally:
        if log_writer is not None:
            log_writer.close()
    print(f'DinoDiT training time: {time.monotonic() - start_time:.1f}s')


if __name__ == '__main__':
    main(get_args_parser().parse_args())
