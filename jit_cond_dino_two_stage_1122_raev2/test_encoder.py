"""测试 DINOv3-L k=7 encoder 是否能正确加载和编码。

在 H200 上运行此脚本验证基本功能。
"""
import sys
from pathlib import Path

import torch

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from stage1.dinov3_k7_encoder import DINOv3Lk7Encoder


def test_encoder():
    print("=== Testing DINOv3-L k=7 Encoder ===\n")

    # 配置（需在 H200 上修改）
    stat_path = "stat/dinov3l_k7_imagenet1k_stats.pt"
    weights_path = "/data0/pretrain_models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    repo = None  # 或指定 DINOv3 源码路径

    print(f"Stat path: {stat_path}")
    print(f"Weights path: {weights_path}")
    print(f"Repo: {repo or 'auto-detect'}\n")

    # 检查文件
    if not Path(stat_path).is_file():
        print(f"❌ Stat file not found: {stat_path}")
        print("Download from: https://huggingface.co/nyu-visionx/RAEv2-models/blob/main/stage1/imagenet/dinov3l-k7/stats.pt")
        return False

    if not Path(weights_path).is_file():
        print(f"❌ Weights not found: {weights_path}")
        return False

    # 创建 encoder
    print("Loading encoder...")
    try:
        encoder = DINOv3Lk7Encoder(
            input_size=256,
            weights_path=weights_path,
            repo=repo,
            stat_path=stat_path,
        )
        print(f"✅ Encoder loaded")
        print(f"   Embed dim: {encoder.embed_dim}")
        print(f"   Patch size: {encoder.patch_size}")
        print(f"   Layer indices: {encoder.LAYER_INDICES}\n")
    except Exception as e:
        print(f"❌ Failed to load encoder: {e}")
        return False

    # 测试编码
    print("Testing encoding...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)

    # 创建随机输入（模拟 [0,1] RGB 图像）
    batch_size = 2
    x = torch.rand(batch_size, 3, 256, 256, device=device)

    try:
        with torch.no_grad():
            latent = encoder.encode(x)
        print(f"✅ Encoding successful")
        print(f"   Input shape: {x.shape}")
        print(f"   Output shape: {latent.shape}")
        print(f"   Output mean: {latent.mean().item():.4f}")
        print(f"   Output std: {latent.std().item():.4f}")
        print(f"   Output range: [{latent.min().item():.4f}, {latent.max().item():.4f}]\n")
    except Exception as e:
        print(f"❌ Encoding failed: {e}")
        return False

    # 验证输出形状
    expected_shape = (batch_size, 1024, 16, 16)
    if latent.shape == expected_shape:
        print(f"✅ Output shape correct: {latent.shape}")
    else:
        print(f"❌ Output shape mismatch: expected {expected_shape}, got {latent.shape}")
        return False

    print("\n=== All tests passed ===")
    return True


if __name__ == "__main__":
    success = test_encoder()
    sys.exit(0 if success else 1)
