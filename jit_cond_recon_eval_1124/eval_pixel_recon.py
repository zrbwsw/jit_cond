"""两阶段重建评估：DinoDiT 重建 DINO 特征，Pixel JiT 重建像素图像。

用法（从本目录运行）：
  python eval_pixel_recon.py \
      --dino_checkpoint  <stage1 ckpt> --dino_weights ema1 \
      --pixel_checkpoint <stage2 ckpt> --pixel_weights ema1 \
      --data_path <imagenet_root> --split val \
      --num_images 2000 --save_images 64 --output_dir eval_out

功能：
  1. 用 DinoDiT（invert→decode）重建 DINO latent，统计 MSE / relative-L2 / patch-cosine。
  2. 用 Pixel JiT（invert→decode）重建 RGB 像素图，统计像素 MSE / PSNR。
     grounding tokens 来自同图 DinoDiT 重建特征，模拟两阶段采样的实际条件。
  3. 保存原图与重建图的并排对比图（PNG）。

所有指标均在 --split 指定集合上计算，默认 val 集。
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets

from checkpoint_io import load_branch, read_checkpoint, use_ema
from denoiser import Denoiser
from stage1.config_dino import add_dino_args, validate_dino_args
from util.training import add_training_args, center_transform, initialize_run, validate_training_args


# ---------------------------------------------------------------------------
# args
# ---------------------------------------------------------------------------

def get_args_parser():
    parser = argparse.ArgumentParser(description="Two-stage pixel reconstruction evaluation")
    add_training_args(parser, epochs=1)
    add_dino_args(parser)
    # Stage 1
    parser.add_argument("--dino_checkpoint", required=True)
    parser.add_argument("--dino_weights", choices=["model", "ema1", "ema2"], default="ema1")
    # Stage 2
    parser.add_argument("--pixel_checkpoint", required=True)
    parser.add_argument("--pixel_weights", choices=["model", "ema1", "ema2"], default="ema1")
    # Pixel sampling
    parser.add_argument("--model", default="JiT-B/16")
    parser.add_argument("--condition_mode_pixel", default="pred",
                        choices=["noised", "pred", "cached_recon"])
    parser.add_argument("--noise_beta_dino_pixel", default=0.25, type=float)
    parser.add_argument("--t_schedule_dino_pixel", default="uniform",
                        choices=["uniform", "logit_normal"])
    parser.add_argument("--t_mean_dino_pixel", default=1.95, type=float)
    parser.add_argument("--t_std_dino_pixel", default=0.5, type=float)
    parser.add_argument("--t_mean_pixel", default=-0.8, type=float)
    parser.add_argument("--t_std_pixel", default=0.8, type=float)
    parser.add_argument("--noise_scale_pixel", default=1.0, type=float)
    parser.add_argument("--steps_pixel", default=50, type=int,
                        help="Heun steps for Pixel JiT invert/decode")
    parser.add_argument("--sampling_method_pixel", default="heun",
                        choices=["euler", "heun"])
    parser.add_argument("--cfg_pixel", default=1.0, type=float,
                        help="CFG scale for Pixel JiT reconstruction; 1.0 = no CFG")
    parser.add_argument("--interval_min_pixel", default=0.1, type=float)
    parser.add_argument("--interval_max_pixel", default=1.0, type=float)
    # Eval options
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--num_images", default=2000, type=int,
                        help="Total images to evaluate across all ranks")
    parser.add_argument("--eval_batch_size", default=8, type=int)
    parser.add_argument("--save_images", default=64, type=int,
                        help="Number of side-by-side comparison images to save (rank 0 only)")
    parser.add_argument("--output_dir", default="eval_pixel_recon_out")
    return parser


def validate_args(args):
    validate_dino_args(args)
    validate_training_args(args)
    if args.num_images < 1 or args.eval_batch_size < 1:
        raise ValueError("num_images and eval_batch_size must be positive")
    if args.save_images < 0:
        raise ValueError("save_images must be non-negative")


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------

def make_loader(args, device):
    folder = datasets.ImageFolder(
        str(Path(args.data_path) / args.split),
        transform=center_transform(args.img_size),
    )
    total = min(args.num_images, len(folder))
    # 每个 rank 取连续不重叠的片段
    from util import misc
    rank, world_size = misc.get_rank(), misc.get_world_size()
    indices = list(range(rank, total, world_size))
    loader = torch.utils.data.DataLoader(
        folder,
        sampler=indices,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    return loader, folder, total


# ---------------------------------------------------------------------------
# image I/O helpers
# ---------------------------------------------------------------------------

def tensor_to_uint8(t):
    """[B,3,H,W] in [-1,1] → uint8 numpy [B,H,W,3] RGB."""
    arr = ((t.float().clamp(-1, 1) + 1) * 127.5).round().byte()
    return arr.permute(0, 2, 3, 1).cpu().numpy()


def save_comparison(orig, recon, out_path, n):
    """Save n pairs as a horizontal grid: orig | recon, stacked vertically."""
    try:
        import cv2
    except ImportError:
        print("cv2 not available; skipping image save", flush=True)
        return
    orig_np = tensor_to_uint8(orig[:n])
    recon_np = tensor_to_uint8(recon[:n])
    rows = []
    for o, r in zip(orig_np, recon_np):
        sep = np.ones((o.shape[0], 4, 3), dtype=np.uint8) * 128
        # cv2 expects BGR
        rows.append(np.concatenate([o[:, :, ::-1], sep, r[:, :, ::-1]], axis=1))
    grid = np.concatenate(rows, axis=0)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)
    print(f"Saved comparison → {out_path}", flush=True)


# ---------------------------------------------------------------------------
# main eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_eval(model, loader, device, args, total_images):
    from util import misc

    out = Path(args.output_dir)
    if misc.is_main_process():
        out.mkdir(parents=True, exist_ok=True)

    # Accumulators: dino_mse, dino_rel, dino_cos, pix_mse, count
    totals = torch.zeros(5, device=device, dtype=torch.float64)

    saved_images = 0
    all_orig, all_recon_pix = [], []

    for batch_idx, (images, labels) in enumerate(loader):
        x = images.to(device).float().div_(127.5).sub_(1)   # [-1,1]
        y = labels.to(device)

        # ---- Stage 1: DINO latent 重建 ----
        clean_dino = model.encode_grounding(x).float()          # [B,768,16,16]
        rec_dino   = model.reconstruct_dino_latent(clean_dino, y).float()

        dino_mse = (rec_dino - clean_dino).square().flatten(1).mean(1)
        dino_rel = (dino_mse / clean_dino.square().flatten(1).mean(1).clamp_min(1e-12)).sqrt()
        dino_cos = F.cosine_similarity(rec_dino, clean_dino, dim=1).flatten(1).mean(1)

        # ---- Stage 2: Pixel JiT 重建（以 DinoDiT 重建特征为 grounding） ----
        grounding_tokens = model._as_tokens(rec_dino)           # [B,256,768]
        t_tag = torch.ones(len(y), device=device)
        # invert → decode，不调用 CFG（评估重建一致性，不做 guided generation）
        rec_pix = model.reconstruct_pixel(x, y, grounding_tokens)  # [B,3,H,W]

        pix_mse = (rec_pix - x).square().flatten(1).mean(1)

        b = len(x)
        totals[0] += dino_mse.sum().double()
        totals[1] += dino_rel.sum().double()
        totals[2] += dino_cos.sum().double()
        totals[3] += pix_mse.sum().double()
        totals[4] += b

        # 收集首批用于保存对比图
        if misc.is_main_process() and saved_images < args.save_images:
            need = args.save_images - saved_images
            all_orig.append(x[:need].cpu())
            all_recon_pix.append(rec_pix[:need].cpu())
            saved_images += min(b, need)

        if batch_idx % 10 == 0:
            sys.stderr.write(
                f"[rank {misc.get_rank()}] {int(totals[4].item())} images done\n"
            )
            sys.stderr.flush()

    # all-reduce across ranks
    if misc.is_dist_avail_and_initialized():
        torch.distributed.all_reduce(totals)
        torch.distributed.barrier()

    count = int(totals[4].item())
    metrics = {
        "dino_mse":        (totals[0] / count).item(),
        "dino_relative_l2":(totals[1] / count).item(),
        "dino_patch_cosine":(totals[2] / count).item(),
        "pixel_mse":       (totals[3] / count).item(),
        "pixel_psnr_db":   float(-10 * np.log10(max((totals[3] / count).item(), 1e-12))),
        "images":          count,
        "split":           args.split,
        "dino_weights":    args.dino_weights,
        "pixel_weights":   args.pixel_weights,
        "steps_dino":      model.steps_dino,
        "steps_pixel":     args.steps_pixel,
    }

    if misc.is_main_process():
        print("\n=== Reconstruction metrics ===")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

        result_path = out / "metrics.json"
        result_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"\nMetrics saved → {result_path}", flush=True)

        if all_orig and args.save_images > 0:
            orig_all  = torch.cat(all_orig,  dim=0)[:args.save_images]
            recon_all = torch.cat(all_recon_pix, dim=0)[:args.save_images]
            save_comparison(orig_all, recon_all, out / "comparison.png", args.save_images)

    return metrics


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main(args):
    validate_args(args)
    args.latent_spec = None   # 评估脚本不校验 latent_spec
    device, _ = initialize_run(args)

    # 构建模型骨架（不需要 pixel checkpoint 的 latent_spec 校验）
    model = Denoiser(args)

    # 加载 Stage 1 DinoDiT
    dino_ckpt = read_checkpoint(args.dino_checkpoint)
    load_branch(model, dino_ckpt, "dino_net", args.dino_weights, args)
    del dino_ckpt

    # 加载 Stage 2 Pixel JiT（直接 load_branch，不走 resume_model）
    pixel_ckpt = read_checkpoint(args.pixel_checkpoint)
    load_branch(model, pixel_ckpt, "net", args.pixel_weights, args)
    del pixel_ckpt

    model.to(device).eval()
    model.dino_net.requires_grad_(False)

    # EMA 初始化为加载的权重本身（use_ema 要求存在 ema_params1）
    params = [p for p in model.parameters() if p.requires_grad]
    model.ema_params1 = [p.clone() for p in params]
    model.ema_params2 = [p.clone() for p in params]

    loader, _, total_images = make_loader(args, device)
    print(f"Evaluating {total_images} images from {args.split} split", flush=True)

    with use_ema(model):
        metrics = run_eval(model, loader, device, args, total_images)

    return metrics


if __name__ == "__main__":
    main(get_args_parser().parse_args())
