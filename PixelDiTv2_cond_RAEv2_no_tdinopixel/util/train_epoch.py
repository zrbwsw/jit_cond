"""JiT batch loop with PixelDiT2's 6255-update linear warmup."""
import math
from contextlib import nullcontext

import torch

from util import misc


def set_step_lr(optimizer, args, step):
    # Matches LinearLR(start_factor=1e-8, end_factor=1, total_iters=6255).
    progress = min(step / args.warmup_steps, 1.0) if args.warmup_steps else 1.0
    lr = args.lr * (1e-8 + (1.0 - 1e-8) * progress)
    for group in optimizer.param_groups:
        group['lr'] = lr
    return lr


def train_one_epoch(model, model_without_ddp, data_loader, optimizer, device, epoch, log_writer, args):
    model.train()
    logger = misc.MetricLogger(delimiter='  ')
    num_batches = len(data_loader)
    if not num_batches:
        raise ValueError('Training data must contain at least one full per-rank batch')
    optimizer.zero_grad(set_to_none=True)
    for batch_index, (images, labels) in enumerate(logger.log_every(data_loader, args.log_freq, f'Epoch [{epoch}]')):
        group_start = batch_index - batch_index % args.accum_steps
        group_size = min(args.accum_steps, num_batches - group_start)
        update = batch_index + 1 == group_start + group_size
        step = optimizer.param_groups[0].get('completed_updates', 0)
        if batch_index == group_start:
            set_step_lr(optimizer, args, step)
        images = images.to(device, dtype=torch.float32, non_blocking=True).div_(127.5).sub_(1)
        labels = labels.to(device, non_blocking=True)
        sync = model.no_sync() if args.distributed and not update else nullcontext()
        with sync:
            with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                loss, metrics = model(images, labels)
            value = loss.item()
            if not math.isfinite(value):
                raise FloatingPointError(f'Non-finite loss at epoch {epoch}, batch {batch_index}: {value}')
            (loss / group_size).backward()
        if update:
            optimizer.step()
            optimizer.param_groups[0]['completed_updates'] = step + 1
            model_without_ddp.update_ema()
            optimizer.zero_grad(set_to_none=True)
        lr = optimizer.param_groups[0]['lr']
        logger.update(loss=value, lr=lr, **metrics)
        if batch_index % args.log_freq == 0:
            means = {k: misc.all_reduce_mean(v) for k, v in {'loss': value, **metrics}.items()}
            if log_writer:
                xaxis = int((epoch + batch_index / num_batches) * 1000)
                for name, mean in means.items():
                    log_writer.add_scalar(f'train_{name}', mean, xaxis)
                log_writer.add_scalar('lr', lr, xaxis)
