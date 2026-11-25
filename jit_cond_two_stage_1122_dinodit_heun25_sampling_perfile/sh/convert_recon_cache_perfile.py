"""Convert v2 shard caches to v3 per-file caches with resumable commits."""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torchvision.datasets import ImageFolder

from util.recon_cache import digest, feature_path


def write_json(path, obj):
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def save_and_verify(row, target):
    if target.exists():
        try:
            existing = np.load(target, allow_pickle=False)
            if existing.shape == row.shape and existing.dtype == row.dtype and np.array_equal(existing, row):
                return
        except (OSError, ValueError):
            pass
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.npy")
    np.save(temporary, row)
    os.replace(temporary, target)
    actual = np.load(target, allow_pickle=False)
    if actual.shape != row.shape or actual.dtype != row.dtype or not np.array_equal(actual, row):
        raise ValueError(f"Per-file verification failed: {target}")


def convert_shard(task):
    source_root, part, image_paths, image_root = task
    source_root = Path(source_root)
    source = source_root / f"part-{part:06d}.npy"
    if source.exists():
        rows = np.load(source, allow_pickle=False)
        if rows.shape != (len(image_paths), 256, 768) or rows.dtype != np.float16:
            raise ValueError(f"Invalid source shard: {source}")
        for offset, image_path in enumerate(image_paths):
            target = feature_path(source_root, image_path, image_root)
            save_and_verify(rows[offset].copy(), target)
    else:
        for image_path in image_paths:
            target = feature_path(source_root, image_path, image_root)
            feature = np.load(target, allow_pickle=False)
            if feature.shape != (256, 768) or feature.dtype != np.float16 or not np.isfinite(feature).all():
                raise ValueError(f"Invalid converted file: {target}")
    return part, len(image_paths)


def perfile_spec(source_spec, count):
    spec = {key: value for key, value in source_spec.items() if key not in {"shard_size", "solver"}}
    spec.update(version=3, layout="per_file", count=count)
    return spec


def migrate_checkpoint(source, project_root, cache_root, spec, cache_id):
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    output_dir = (project_root / "checkpoints" /
                  "stage1-dinov3b-dinodit-b-recon-noflip-e80-stage2-pixeljit-b-dinodit-cached-recon-perfile-e320")
    output_dir.mkdir(parents=True, exist_ok=True)
    args = checkpoint["args"]
    args.cache_id = cache_id
    args.expected_cache_id = cache_id
    args.grounding_cache = str(cache_root)
    args.output_dir = str(output_dir)
    args.cache_cfg_dino = 1.0
    args.cfg_dino = 2.0
    output = output_dir / "checkpoint-last-perfile.pth"
    temporary = output_dir / "checkpoint-last-perfile.tmp.pth"
    torch.save(checkpoint, temporary)
    os.replace(temporary, output)
    print(f"Wrote resumable per-file checkpoint: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_root", required=True)
    parser.add_argument("--data_path", default="/mnt/shared-storage-user/datafactory-share/gushishen/imagenet")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch_shards", type=int, default=50)
    parser.add_argument("--limit_images", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--migrate_checkpoint", default="")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.batch_shards < 1:
        raise ValueError("workers and batch_shards must be positive")

    root = Path(args.source_root)
    source_spec = json.loads((root / "spec.json").read_text(encoding="utf-8"))
    if source_spec.get("version") != 2 or source_spec.get("shard_size") != 128:
        raise ValueError("Expected a v2 cache with 128 rows per shard")
    folder = ImageFolder(str(Path(args.data_path) / source_spec["split"]))
    image_root = Path(folder.root)
    count = len(folder) if not args.limit_images else min(args.limit_images, len(folder))
    samples = folder.samples[:count]
    total_shards = (count + 127) // 128
    progress_path = root / "migration_progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    next_part = progress.get("next_part", 0)
    spec = perfile_spec(source_spec, count)
    cache_id = digest([spec, count])
    if args.dry_run:
        print(f"source={root}; next_part={next_part}; total_shards={total_shards}; "
              f"workers={args.workers}; batch_shards={args.batch_shards}; cache_id={cache_id}")
        return

    if args.max_batches < 0:
        raise ValueError("max_batches must be non-negative")
    started = time.monotonic()
    batches = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        while next_part < total_shards:
            batch_end = min(next_part + args.batch_shards, total_shards)
            tasks = []
            for part in range(next_part, batch_end):
                image_paths = [samples[index][0] for index in range(part * 128, min((part + 1) * 128, count))]
                tasks.append((str(root), part, image_paths, str(image_root)))
            for part, rows in executor.map(convert_shard, tasks):
                (root / f"part-{part:06d}.npy").unlink(missing_ok=True)
                (root / f"part-{part:06d}.sha256").unlink(missing_ok=True)
                next_part = part + 1
                write_json(progress_path, dict(next_part=next_part, total_shards=total_shards,
                                               complete=False, cache_id=cache_id))
            elapsed = time.monotonic() - started
            converted = min(next_part * 128, count)
            rate = converted / elapsed if elapsed > 0 else 0
            print(f"Committed shards {batch_end}/{total_shards}; images={converted}/{count}; "
                  f"rate={rate:.1f} images/s; elapsed={elapsed / 60:.1f} min", flush=True)
            batches += 1
            if args.max_batches and batches >= args.max_batches:
                break
    if next_part < total_shards:
        print(f"Paused after {batches} batches; next_part={next_part}", flush=True)
        return

    write_json(root / "spec.json", spec)
    write_json(root / "manifest.json", dict(spec=spec, count=count, id=cache_id))
    write_json(progress_path, dict(next_part=total_shards, total_shards=total_shards,
                                   complete=True, cache_id=cache_id))
    print(f"Per-file cache complete: {count} images, cache_id={cache_id}", flush=True)
    if args.migrate_checkpoint:
        project_root = Path(__file__).resolve().parents[1]
        migrate_checkpoint(args.migrate_checkpoint, project_root, root, spec, cache_id)


if __name__ == "__main__":
    main()
