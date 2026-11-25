"""Export per-file grounding cache (version 4: free sampling).

每张训练图对应一个 DinoDiT 自由采样的 DINO latent（从随机噪声 25步 Heun），
路径结构镜像 ImageNet train 目录：
  <cache_dir>/<synset>/<stem>.npy   shape=(256,768) dtype=float16

与 version 3（重建）的区别：
  - 重建（v3）：encode clean image → invert → decode，保留实例信息
  - 采样（v4）：从 N(0,σ²I) 随机初始化 → 25步 Heun + CFG，每图独立生成

Stage 2 训练时读取这些采样特征，推理时同样自由采样，消除 train-test gap。

运行方式（从项目根目录）：
  torchrun --nproc_per_node=8 -m stage1.export_recon \
      --dino_checkpoint <ckpt> --dino_weights ema1 \
      --cache_dir <dir> --split train \
      --steps_dino 25 --cfg_dino 2.0 \
      --seed_base 42
"""
import json
import os
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import ImageFolder

from checkpoint_io import latent_spec, load_branch, read_checkpoint
from stage1.denoiser_dino import DinoLatentModel
from stage1.main_dino import get_args_parser, validate_args
from util.recon_cache import (
    dataset_hash, digest, feature_path, file_hash, teacher_hash,
)
from util.training import center_transform


def write_json(path, obj):
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def export_rank(args, rank, local_rank, world_size):
    if local_rank >= 0:
        torch.cuda.set_device(local_rank)
        args.device = f"cuda:{local_rank}"

    root = Path(args.cache_dir)
    root.mkdir(parents=True, exist_ok=True)

    folder = ImageFolder(
        str(Path(args.data_path) / args.split),
        transform=center_transform(args.img_size),
    )
    image_root = Path(folder.root)

    model = DinoLatentModel(args)
    ckpt = read_checkpoint(args.dino_checkpoint)
    load_branch(model, ckpt, "dino_net", args.dino_weights, args)
    del ckpt
    model.to(args.device).eval()

    spec = dict(
        version=4,
        split=args.split,
        count=len(folder),
        dataset=dataset_hash(folder),
        teacher=teacher_hash(model),
        checkpoint_sha256=file_hash(args.dino_checkpoint),
        weights=args.dino_weights,
        latent=latent_spec(args),
        size=args.img_size,
        steps=args.steps_dino,
        method=args.sampling_method_dino,
        cfg=args.cfg_dino,
        noise_scale=args.noise_scale_dino,
        interval=[args.interval_min_dino, args.interval_max_dino],
        t_eps=args.t_eps,
        transform="adm_center_noflip",
        layout="per_file_sampling",
        dtype="float16",
        seed_base=args.seed_base,
    )

    # rank 0 写 spec，检查一致性
    spec_path = root / "spec.json"
    if rank == 0:
        if spec_path.exists():
            existing = json.loads(spec_path.read_text(encoding="utf-8"))
            if existing != spec:
                raise ValueError("Different cache recipe already exists in this directory")
        write_json(spec_path, spec)

    if world_size > 1:
        torch.distributed.barrier()

    # 每个 rank 负责交错的索引子集；dataset 直接返回全局索引，避免从批序号反推。
    class IndexedFolder(torch.utils.data.Dataset):
        def __init__(self, folder):
            self.folder = folder

        def __len__(self):
            return len(self.folder)

        def __getitem__(self, idx):
            image, label = self.folder[idx]
            return idx, image, label

    indices = list(range(rank, len(folder), world_size))
    loader = torch.utils.data.DataLoader(
        IndexedFolder(folder),
        sampler=indices,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    done = 0
    for batch_indices, images, labels in loader:
        batch_indices = batch_indices.tolist()
        batch_labels = labels.to(args.device)

        # ===== 修改点：从随机噪声采样，每图独立种子 =====
        # 不再使用 clean image 重建，改为自由采样 p(z|y)
        bsz = len(batch_labels)
        sampled_latents = []
        for i, idx in enumerate(batch_indices):
            # 每图独立种子 = seed_base + 全局索引，确保可复现且不重复
            seed = args.seed_base + idx
            torch.manual_seed(seed)
            if args.device.startswith("cuda"):
                torch.cuda.manual_seed_all(seed)

            # 单个样本的 DinoDiT 自由采样（25步 Heun + CFG）
            y_single = batch_labels[i:i+1]
            z_sampled = model.sample_dino_latent(y_single)  # [1, 768, 16, 16]
            sampled_latents.append(z_sampled)

        sampled_latents = torch.cat(sampled_latents, dim=0)  # [B, 768, 16, 16]
        tokens = model._as_tokens(sampled_latents).half()   # [B, 256, 768] float16
        # ===== 修改结束 =====

        if not tokens.isfinite().all():
            raise ValueError(f"Non-finite cache output at rank {rank} batch {done}")

        tokens_np = tokens.cpu().numpy()
        for i, idx in enumerate(batch_indices):
            img_path, _ = folder.samples[idx]
            npy_path = feature_path(root, img_path, image_root)
            npy_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = npy_path.with_suffix(".tmp.npy")
            np.save(str(tmp), tokens_np[i])
            os.replace(tmp, npy_path)

        done += 1
        if done % 50 == 0:
            print(f"[rank {rank}] {done * args.batch_size} batches (~"
                  f"{rank + done * args.batch_size * world_size} / {len(folder)} images)",
                  flush=True)

    if world_size > 1:
        torch.distributed.barrier()

    # rank 0 写 manifest
    if rank == 0:
        cache_id = digest([spec, len(folder)])
        write_json(root / "manifest.json", dict(
            spec=spec,
            count=len(folder),
            id=cache_id,
        ))
        print(f"Cache complete: {len(folder)} files → {root}", flush=True)


@torch.inference_mode()
def export(args):
    validate_args(args)
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    try:
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
    parser.add_argument("--seed_base", type=int, default=42,
                        help="Base seed for per-image sampling; actual seed = seed_base + global_idx")
    export(parser.parse_args())
