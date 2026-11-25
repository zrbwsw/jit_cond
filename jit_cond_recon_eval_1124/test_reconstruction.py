"""测试 DinoDiT 和 Pixel JiT 的 invert→decode 重建逻辑。

从 ImageNet 训练集随机采样图像，可视化：
1. 原图
2. DinoDiT latent 加噪过程（0 → 1，每隔 0.2 保存一帧）
3. DinoDiT latent 重建结果（invert → decode）
4. Pixel 图像加噪过程
5. Pixel 图像重建结果（invert → decode）

验证重建逻辑是否正确，以及模型输出是速度场 v(z,t)。
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import functional as TF

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from checkpoint_io import load_branch, read_checkpoint
from denoiser import Denoiser
from stage1.config_dino import add_dino_args
from util.training import add_training_args, center_transform


def tensor_to_image(t):
    """[C,H,W] in [-1,1] → [H,W,3] uint8 RGB."""
    arr = ((t.clamp(-1, 1) + 1) * 127.5).round().byte()
    return arr.permute(1, 2, 0).cpu().numpy()


def save_grid(images, path, ncol=6):
    """保存图像网格，images 是 list of [H,W,3] uint8 numpy。"""
    import cv2
    nrow = (len(images) + ncol - 1) // ncol
    # 补齐到完整网格
    while len(images) < nrow * ncol:
        images.append(np.zeros_like(images[0]))

    rows = []
    for r in range(nrow):
        row_imgs = images[r * ncol : (r + 1) * ncol]
        rows.append(np.concatenate(row_imgs, axis=1))
    grid = np.concatenate(rows, axis=0)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), grid[:, :, ::-1])  # RGB → BGR
    print(f"Saved → {path}")


@torch.no_grad()
def test_reconstruction(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据集
    folder = datasets.ImageFolder(
        str(Path(args.data_path) / "train"),
        transform=center_transform(args.img_size),
    )
    print(f"Dataset: {len(folder)} images")

    # 随机采样
    torch.manual_seed(args.seed)
    indices = torch.randperm(len(folder))[:args.num_samples].tolist()

    images_list, labels_list = [], []
    for idx in indices:
        img, label = folder[idx]
        images_list.append(img)
        labels_list.append(label)

    images = torch.stack(images_list).to(device).float().div_(127.5).sub_(1)  # [N,3,H,W] in [-1,1]
    labels = torch.tensor(labels_list, device=device)
    print(f"Sampled {len(images)} images, labels: {labels.tolist()}")

    # 构建模型
    model = Denoiser(args)

    # 加载 DinoDiT
    dino_ckpt = read_checkpoint(args.dino_checkpoint)
    load_branch(model, dino_ckpt, "dino_net", args.dino_weights, args)
    del dino_ckpt

    # 加载 Pixel JiT
    pixel_ckpt = read_checkpoint(args.pixel_checkpoint)
    load_branch(model, pixel_ckpt, "net", args.pixel_weights, args)
    del pixel_ckpt

    model.to(device).eval()
    print(f"Model loaded: DinoDiT {args.steps_dino} steps, Pixel {args.steps_pixel} steps")

    # ========== Part 1: DinoDiT latent 重建 ==========
    print("\n=== DinoDiT latent reconstruction ===")

    # Encode to DINO latent
    z_clean = model.encode_grounding(images)  # [N, 768, 16, 16]
    print(f"Clean latent shape: {z_clean.shape}")

    # 加噪过程可视化（从 t=0 到 t=1）
    t_snapshots = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    noised_latents = []
    for t in t_snapshots:
        t_tensor = torch.full((len(images),), t, device=device).view(-1, 1, 1, 1)
        z_noised = t_tensor * z_clean + (1 - t_tensor) * torch.randn_like(z_clean) * model.noise_scale_dino
        noised_latents.append(z_noised)

    # Invert → Decode 重建
    z_inverted = model.invert_dino_latent(z_clean, labels)
    z_reconstructed = model.decode_dino_latent(z_inverted, labels)

    # 计算重建误差
    mse = (z_reconstructed - z_clean).square().flatten(1).mean(1)
    cos_sim = F.cosine_similarity(z_reconstructed, z_clean, dim=1).flatten(1).mean(1)
    print(f"DinoDiT reconstruction MSE: {mse.mean().item():.6f}")
    print(f"DinoDiT reconstruction cosine: {cos_sim.mean().item():.6f}")

    # ========== Part 2: Pixel 图像重建 ==========
    print("\n=== Pixel image reconstruction ===")

    # 以 DinoDiT 重建 latent 为 grounding
    grounding_tokens = model._as_tokens(z_reconstructed)

    # 图像加噪过程可视化
    noised_images_list = []
    for t in t_snapshots:
        t_tensor = torch.full((len(images),), t, device=device).view(-1, 1, 1, 1)
        x_noised = t_tensor * images + (1 - t_tensor) * torch.randn_like(images) * model.noise_scale_pixel
        noised_images_list.append(x_noised)

    # Pixel 重建
    x_reconstructed = model.reconstruct_pixel(images, labels, grounding_tokens, steps=args.steps_pixel)

    # 计算重建误差
    pixel_mse = (x_reconstructed - images).square().flatten(1).mean(1)
    pixel_psnr = -10 * torch.log10(pixel_mse.clamp_min(1e-10))
    print(f"Pixel reconstruction MSE: {pixel_mse.mean().item():.6f}")
    print(f"Pixel reconstruction PSNR: {pixel_psnr.mean().item():.2f} dB")

    # ========== 保存可视化 ==========
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 原图
    orig_imgs = [tensor_to_image(images[i]) for i in range(len(images))]
    save_grid(orig_imgs, out_dir / "1_original.png")

    # 2. 图像加噪过程（6 个时间点）
    noising_grid = []
    for t_idx, t in enumerate(t_snapshots):
        for i in range(len(images)):
            noising_grid.append(tensor_to_image(noised_images_list[t_idx][i]))
    save_grid(noising_grid, out_dir / "2_image_noising_process.png", ncol=len(images))

    # 3. Pixel 重建结果
    recon_imgs = [tensor_to_image(x_reconstructed[i]) for i in range(len(images))]
    save_grid(recon_imgs, out_dir / "3_pixel_reconstructed.png")

    # 4. 原图 vs 重建（并排对比）
    comparison = []
    for i in range(len(images)):
        comparison.append(orig_imgs[i])
        comparison.append(recon_imgs[i])
    save_grid(comparison, out_dir / "4_original_vs_reconstructed.png", ncol=4)

    # 5. 保存指标
    metrics = {
        "dino_mse": mse.tolist(),
        "dino_cosine": cos_sim.tolist(),
        "pixel_mse": pixel_mse.tolist(),
        "pixel_psnr_db": pixel_psnr.tolist(),
        "config": {
            "steps_dino": args.steps_dino,
            "steps_pixel": args.steps_pixel,
            "noise_scale_dino": model.noise_scale_dino,
            "noise_scale_pixel": model.noise_scale_pixel,
            "method_dino": model.method_dino,
            "method_pixel": model.method_pixel,
        }
    }
    import json
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nMetrics saved to {out_dir / 'metrics.json'}")

    print(f"\n=== Test complete ===")
    print(f"Output directory: {out_dir}")
    print(f"Files:")
    print(f"  1_original.png              - 原图")
    print(f"  2_image_noising_process.png - 加噪过程 (t=0,0.2,0.4,0.6,0.8,1.0)")
    print(f"  3_pixel_reconstructed.png   - 重建结果")
    print(f"  4_original_vs_reconstructed.png - 原图 vs 重建对比")
    print(f"  metrics.json                - 重建指标")


def get_args():
    parser = argparse.ArgumentParser(description="Test reconstruction logic")
    add_training_args(parser, epochs=1)
    add_dino_args(parser)

    parser.add_argument("--dino_checkpoint", required=True)
    parser.add_argument("--dino_weights", choices=["model", "ema1", "ema2"], default="ema1")
    parser.add_argument("--pixel_checkpoint", required=True)
    parser.add_argument("--pixel_weights", choices=["model", "ema1", "ema2"], default="ema1")

    parser.add_argument("--model", default="JiT-B/16")
    parser.add_argument("--condition_mode_pixel", default="pred")  # 不影响重建测试

    # Pixel sampling 参数（重建用）
    parser.add_argument("--steps_pixel", default=50, type=int)
    parser.add_argument("--sampling_method_pixel", choices=["euler", "heun"], default="heun")
    parser.add_argument("--cfg_pixel", default=1.0, type=float)
    parser.add_argument("--noise_scale_pixel", default=1.0, type=float)
    parser.add_argument("--t_mean_pixel", default=-0.8, type=float)
    parser.add_argument("--t_std_pixel", default=0.8, type=float)
    parser.add_argument("--interval_min_pixel", default=0.1, type=float)
    parser.add_argument("--interval_max_pixel", default=1.0, type=float)

    parser.add_argument("--num_samples", type=int, default=8, help="Number of test images")

    args = parser.parse_args()

    # seed 和 output_dir 由 training args 提供，设置默认值
    if not hasattr(args, 'seed') or args.seed is None:
        args.seed = 42
    if not args.output_dir:
        args.output_dir = "test_reconstruction_output"

    return args


if __name__ == "__main__":
    args = get_args()
    args.latent_spec = None  # 测试脚本不校验
    test_reconstruction(args)
