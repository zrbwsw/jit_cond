"""Full validation pass in normalized DINO patch space with the cache/export teacher."""
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import datasets

from checkpoint_io import use_ema
from util import misc
from util.training import center_transform


def make_val_loader(args, class_to_idx):
    folder = datasets.ImageFolder(str(Path(args.data_path) / "val"), transform=center_transform(args.img_size))
    if folder.class_to_idx != class_to_idx:
        raise ValueError("Validation class_to_idx differs from the training set")
    if len(folder) != args.recon_val_expected:
        raise ValueError(f"Expected {args.recon_val_expected} validation images, found {len(folder)}")
    # One owner per index, even when the dataset or last batch does not divide evenly.
    indices = range(misc.get_rank(), len(folder), misc.get_world_size())
    return torch.utils.data.DataLoader(
        folder, sampler=indices, batch_size=args.recon_eval_batch, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False, generator=torch.Generator().manual_seed(0))


@torch.no_grad()
def evaluate_recon(model, loader, device, epoch, output_dir, writer=None):
    # All ranks run the bare model: unequal numbers of batches require no per-batch collective.
    started = time.monotonic()
    previous_mode = model.training
    totals = torch.zeros(4, device=device, dtype=torch.float64)
    model.eval()
    try:
        with use_ema(model):
            for images, labels in loader:
                x = images.to(device, non_blocking=True).float().div_(127.5).sub_(1)
                y = labels.to(device, non_blocking=True)
                clean = model.encode_grounding(x).float()
                rec = model.reconstruct_dino_latent(clean, y).float()
                mse = (rec - clean).square().flatten(1).mean(1)
                relative = (mse / clean.square().flatten(1).mean(1).clamp_min(1e-12)).sqrt()
                cosine = F.cosine_similarity(rec, clean, dim=1).flatten(1).mean(1)
                totals[:3] += torch.stack([mse, relative, cosine]).sum(dim=1, dtype=torch.float64)
                totals[3] += len(x)
    finally:
        model.train(previous_mode)
    rank, world_size = misc.get_rank(), misc.get_world_size()
    local_count = int(totals[3].item())
    local_expected = len(range(rank, len(loader.dataset), world_size))
    local_finite = bool(torch.isfinite(totals).all().item())
    sys.stderr.write(
        f"[rank {rank}] Local reconstruction validation epoch {epoch}: "
        f"{local_count}/{local_expected} images, finite={local_finite}, "
        f"elapsed={time.monotonic() - started:.1f}s\n")
    sys.stderr.flush()
    if misc.is_dist_avail_and_initialized():
        torch.distributed.all_reduce(totals)
        torch.distributed.barrier()
    count = int(totals[3].item())
    if count != len(loader.dataset) or not torch.isfinite(totals).all():
        raise RuntimeError(f"Incomplete or non-finite validation at epoch {epoch}: {totals.tolist()}")
    metrics = dict(zip(("mse", "relative_l2", "patch_cosine"), (totals[:3] / count).tolist()))
    record = dict(epoch=epoch, images=count, weights="ema1", steps=model.steps_dino,
                  method=model.method_dino, cfg=model.cfg_dino, **metrics)
    if misc.is_main_process():
        print("Stage 1 validation " + json.dumps(record), flush=True)
        output = Path(output_dir) / f"val-recon-e{epoch:04d}.json"
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
        os.replace(temporary, output)
        if writer is not None:
            for key, value in metrics.items():
                writer.add_scalar("val_recon/" + key, value, epoch)
            writer.add_scalar("val_recon/images", count, epoch)
    return record
