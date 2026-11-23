"""Export paired DINO reconstructions. Run from the project root with python -m."""
import json
import os
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock
from torchvision.datasets import ImageFolder

from checkpoint_io import latent_spec, load_branch, read_checkpoint
from stage1.denoiser_dino import DinoLatentModel
from stage1.main_dino import get_args_parser, validate_args
from util.recon_cache import dataset_hash, digest, file_hash, shard_path, teacher_hash
from util.training import center_transform


def write_json(path, obj):
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def rank_batches(folder, rank, world_size, shard_size):
    count = (len(folder) + shard_size - 1) // shard_size
    batches = []
    for part in range(rank, count, world_size):
        first = part * shard_size
        rows = min(shard_size, len(folder) - first)
        batches.append(list(range(first, first + rows)))
    return batches


def export_rank(args, rank, local_rank, world_size):
    if local_rank >= 0:
        torch.cuda.set_device(local_rank)
        args.device = f"cuda:{local_rank}"
    root = Path(args.cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    folder = ImageFolder(str(Path(args.data_path) / args.split), transform=center_transform(args.img_size))
    model = DinoLatentModel(args)
    ckpt = read_checkpoint(args.dino_checkpoint)
    load_branch(model, ckpt, "dino_net", args.dino_weights, args)
    del ckpt
    spec = dict(version=2, split=args.split, count=len(folder), dataset=dataset_hash(folder),
                teacher=teacher_hash(model), checkpoint_sha256=file_hash(args.dino_checkpoint),
                weights=args.dino_weights, latent=latent_spec(args), size=args.img_size,
                shard_size=args.shard_size, steps=args.steps_dino,
                method=args.sampling_method_dino, cfg=args.cfg_dino, noise_scale=args.noise_scale_dino,
                interval=[args.interval_min_dino, args.interval_max_dino], t_eps=args.t_eps,
                transform="adm_center_noflip", solver="interior_terminal_v1",
                dtype="float16", layout="row,HW,C")
    config = root / "spec.json"
    if config.exists() and json.loads(config.read_text(encoding="utf-8")) != spec:
        raise ValueError("Different cache recipe in this directory")
    if rank == 0:
        write_json(config, spec)
        (root / "manifest.json").unlink(missing_ok=True)
    model.to(args.device).eval()

    receipt = {}
    batches = rank_batches(folder, rank, world_size, args.shard_size)
    loader = torch.utils.data.DataLoader(
        folder, batch_sampler=batches, num_workers=args.num_workers, pin_memory=args.pin_mem)
    for batch_index, (images, labels) in enumerate(loader):
        part = rank + batch_index * world_size
        first = part * args.shard_size
        path = shard_path(root, part)
        part_receipt = path.with_suffix(".sha256")
        if path.exists() and part_receipt.exists() and part_receipt.read_text() == file_hash(path):
            receipt[str(part)] = part_receipt.read_text()
            continue
        rows = len(images)
        tmp = path.with_suffix(".tmp.npy")
        data = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16,
                                         shape=(rows, 256, 768))
        for off in range(0, rows, args.batch_size):
            rgb = images[off:off + args.batch_size].to(args.device).float().div_(127.5).sub_(1)
            batch_labels = labels[off:off + args.batch_size].to(args.device)
            clean = model.encode_grounding(rgb)
            rec = model.reconstruct_dino_latent(clean, batch_labels)
            tokens = model._as_tokens(rec).half()
            if not tokens.isfinite().all():
                raise ValueError(f"Non-finite cache output at row {first + off}")
            data[off:off + len(rgb)] = tokens.cpu().numpy()
        data.flush()
        del data
        os.replace(tmp, path)
        checksum = file_hash(path)
        part_receipt.write_text(checksum)
        receipt[str(part)] = checksum
        print(f"[rank {rank}] Completed {first + rows}/{len(folder)}", flush=True)

    receipts = [None] * world_size if rank == 0 else None
    if world_size > 1:
        torch.distributed.gather_object(receipt, receipts, dst=0)
    else:
        receipts = [receipt]
    if rank != 0:
        return
    count = (len(folder) + args.shard_size - 1) // args.shard_size
    checksums = []
    for part in range(count):
        matches = [receipt[str(part)] for receipt in receipts if str(part) in receipt]
        if len(matches) != 1:
            raise RuntimeError(f"Expected exactly one checksum for part {part}, found {len(matches)}")
        checksums.append(matches[0])
    write_json(root / "manifest.json", dict(spec=spec, shards=checksums, id=digest([spec, checksums])))


@torch.inference_mode()
def export(args):
    validate_args(args)
    if min(args.shard_size, args.batch_size) < 1:
        raise ValueError("Expected positive shard and batch sizes")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError(f"Invalid rank {rank} for world size {world_size}")
    if world_size > 1:
        torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
        if rank == 0:
            with FileLock(str(Path(args.cache_dir) / ".export.lock"), timeout=0):
                export_rank(args, rank, local_rank, world_size)
        else:
            export_rank(args, rank, local_rank, world_size)
    finally:
        if world_size > 1:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = get_args_parser()
    parser.set_defaults(batch_size=8)
    parser.add_argument("--dino_checkpoint", required=True)
    parser.add_argument("--dino_weights", choices=["model", "ema1", "ema2"], default="ema1")
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--shard_size", type=int, default=128)
    export(parser.parse_args())
