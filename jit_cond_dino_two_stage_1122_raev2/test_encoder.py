"""Check the local DINOv3-L k=7 encoder with a small real forward."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from stage1.dinov3_k7_encoder import DINOv3Lk7Encoder


def run_encoder_check():
    project = Path(__file__).parent
    stat_path = project / "stat/dinov3l_k7_imagenet1k_stats.pt"
    weights_path = Path("/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/pretrain_models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
    repo = Path("/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/Ours_BaseT_rmsnorm2_Residual_1016/SSL_code/dinov3")
    for path in (stat_path, weights_path, repo / "hubconf.py"):
        if not path.is_file():
            raise FileNotFoundError(path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = DINOv3Lk7Encoder(
        input_size=256,
        weights_path=str(weights_path),
        repo=str(repo),
        stat_path=str(stat_path),
    ).to(device).eval()
    x = torch.rand(2, 3, 256, 256, device=device)
    with torch.no_grad():
        latent = encoder.encode(x)
    assert latent.shape == (2, 1024, 16, 16), latent.shape
    assert torch.isfinite(latent).all()
    print(f"PASS: latent shape={latent.shape}")
    print(f"mean={latent.mean().item():.4f}; std={latent.std().item():.4f}")
    print(f"range=[{latent.min().item():.4f}, {latent.max().item():.4f}]")


if __name__ == "__main__":
    run_encoder_check()
