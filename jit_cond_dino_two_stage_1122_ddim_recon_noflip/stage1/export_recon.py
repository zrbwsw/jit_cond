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
from util.recon_cache import dataset_hash, digest, file_hash, image_view, shard_path, teacher_hash


def write_json(path, obj):
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


@torch.inference_mode()
def export(args):
    validate_args(args)
    if min(args.shard_size, args.batch_size) < 1:
        raise ValueError("Expected positive shard and batch sizes")
    root = Path(args.cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    folder = ImageFolder(str(Path(args.data_path) / args.split))
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
    model.to(args.device).eval()
    with FileLock(str(root / ".export.lock"), timeout=0):
        config = root / "spec.json"
        if config.exists() and json.loads(config.read_text(encoding="utf-8")) != spec:
            raise ValueError("Different cache recipe in this directory")
        write_json(config, spec)
        checksums = []
        for part, first in enumerate(range(0, len(folder), args.shard_size)):
            path = shard_path(root, part)
            receipt = path.with_suffix(".sha256")
            if path.exists() and receipt.exists() and receipt.read_text() == file_hash(path):
                checksums.append(receipt.read_text())
                continue
            (root / "manifest.json").unlink(missing_ok=True)
            rows = min(args.shard_size, len(folder) - first)
            tmp = path.with_suffix(".tmp.npy")
            data = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16,
                                             shape=(rows, 256, 768))
            for off in range(0, rows, args.batch_size):
                batch = [image_view(folder, first + i, args.img_size)
                         for i in range(off, min(off + args.batch_size, rows))]
                images, labels = zip(*batch)
                rgb = torch.stack(images).to(args.device).float().div_(127.5).sub_(1)
                clean = model.encode_grounding(rgb)
                rec = model.reconstruct_dino_latent(clean, torch.tensor(labels, device=args.device))
                tokens = model._as_tokens(rec).half()
                if not tokens.isfinite().all():
                    raise ValueError(f"Non-finite cache output at row {first + off}")
                data[off:off + len(batch)] = tokens.cpu().numpy()
            data.flush()
            del data
            os.replace(tmp, path)
            checksum = file_hash(path)
            receipt.write_text(checksum)
            checksums.append(checksum)
            print(f"Completed {first + rows}/{len(folder)}", flush=True)
        write_json(root / "manifest.json", dict(spec=spec, shards=checksums, id=digest([spec, checksums])))


if __name__ == "__main__":
    parser = get_args_parser()
    parser.set_defaults(batch_size=8)
    parser.add_argument("--dino_checkpoint", required=True)
    parser.add_argument("--dino_weights", choices=["model", "ema1", "ema2"], default="ema1")
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--shard_size", type=int, default=128)
    export(parser.parse_args())
