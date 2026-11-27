"""Distributed JiT image generation and FID/IS from the Pixel EMA."""
from pathlib import Path
import shutil
import tempfile

import torch
from PIL import Image

from checkpoint_io import use_ema
from util import misc


def _sample_folder(args, epoch):
    # Each evaluation gets an empty shared folder, including repeated CFG sweeps.
    value = [None]
    if misc.is_main_process():
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        value[0] = tempfile.mkdtemp(prefix=f'epoch{epoch}-cfg{args.cfg_pixel}-', dir=output)
    if misc.is_dist_avail_and_initialized():
        torch.distributed.broadcast_object_list(value)
    return Path(value[0])


def _score(folder, args, device):
    import torch_fidelity
    stats = Path(args.fid_stats)
    weights = stats.with_name('weights-inception-2015-12-05-6726825d.pth')
    metrics_options = {'feature_extractor_weights_path': str(weights)} if weights.is_file() else {}
    metrics = torch_fidelity.calculate_metrics(
        input1=str(folder), input2=None, fid_statistics_file=str(stats), cuda=device.type == 'cuda',
        isc=True, fid=True, kid=False, prc=False, verbose=False, **metrics_options)
    return metrics['frechet_inception_distance'], metrics['inception_score_mean']


@torch.no_grad()
def evaluate(model, args, epoch, batch_size, log_writer=None):
    folder = _sample_folder(args, epoch)
    rank, world = misc.get_rank(), misc.get_world_size()
    device = next(model.parameters()).device
    previous = model.training
    model.eval()
    try:
        # Keep training RNG on every rank and make a repeated evaluation reproducible.
        devices = [torch.cuda.current_device()] if device.type == 'cuda' else []
        with use_ema(model), torch.random.fork_rng(devices=devices):
            torch.manual_seed(args.seed + rank)
            for start in range(rank * batch_size, args.num_images, world * batch_size):
                stop = min(start + batch_size, args.num_images)
                indices = torch.arange(start, stop, device=device)
                labels = indices * args.class_num // args.num_images
                with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                    images = model.generate(labels)
                images = ((images.float() + 1) * 127.5).clamp_(0, 255).round_().byte()
                pixels = images.permute(0, 2, 3, 1).contiguous().cpu().numpy()
                for index, pixel in enumerate(pixels, start):
                    Image.fromarray(pixel).save(folder / f'{index:05d}.png')
    finally:
        model.release_latent_sampler()
        model.train(previous)
    if misc.is_dist_avail_and_initialized():
        torch.distributed.barrier()
    if misc.is_main_process():
        if args.samples_only:
            print(f'Saved {args.num_images} images to {folder}')
        else:
            fid, score = _score(folder, args, device)
            print(f'FID: {fid:.4f}, Inception Score: {score:.4f}', flush=True)
            if log_writer:
                log_writer.add_scalar(f'fid_cfg_pixel{args.cfg_pixel}', fid, epoch)
                log_writer.add_scalar(f'is_cfg_pixel{args.cfg_pixel}', score, epoch)
            shutil.rmtree(folder)
    if misc.is_dist_avail_and_initialized():
        torch.distributed.barrier()
