"""Frozen DINOv3 patch-token extraction for JiT conditional AdaLN."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dinov3_backbone import load_dino_v3


class DINOv3GroundingEncoder(nn.Module):
    """提取 clean image 的 DINOv3 patch token，且不向 DINO 反传梯度。

    JiT 图像范围是 ``[-1,1]``，DINOv3 需要 ImageNet 标准化后的图像；
    预处理统一放在这里，使训练和可选的 conditional sampling 使用同一套流程。
    """

    def __init__(
        self,
        model_name: str = "dinov3_vitb16",
        weights_path: Optional[str] = None,
        repo: Optional[str] = None,
    ):
        super().__init__()
        self.model = load_dino_v3(model_name, weights_path=weights_path, repo=repo)
        self.model.requires_grad_(False)
        self.model.eval()
        self.embed_dim = int(getattr(self.model, "embed_dim", getattr(self.model, "num_features", 768)))
        self.patch_size = int(getattr(self.model, "patch_size", 16))
        self.register_buffer(
            "image_mean", torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std", torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def train(self, mode: bool = True):
        # 某些 DINOv3 版本在 train 模式下会使用随机位置行为；作为冻结
        # teacher 时必须始终保持 eval，保证训练和采样的 token 稳定。
        super().train(False)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(self, clean_images: torch.Tensor) -> torch.Tensor:
        if clean_images.ndim != 4 or clean_images.shape[1] != 3:
            raise ValueError(f"expected clean images [B,3,H,W], got {tuple(clean_images.shape)}")
        # JiT 图像范围是 [-1,1]，先还原到 [0,1]，再做 DINOv3 的
        # ImageNet mean/std 标准化。
        x = clean_images.float().add(1.0).mul(0.5)
        x = (x - self.image_mean.to(device=x.device)) / self.image_std.to(device=x.device)
        h, w = x.shape[-2:]
        target_h = max(self.patch_size, (h // self.patch_size) * self.patch_size)
        target_w = max(self.patch_size, (w // self.patch_size) * self.patch_size)
        # DINOv3 patch size 为 16；非整除分辨率先裁到可分 patch 的网格。
        if (target_h, target_w) != (h, w):
            x = F.interpolate(x, size=(target_h, target_w), mode="bicubic", align_corners=False)
        features = self.model.forward_features(x)
        # 去掉 CLS/register token，只保留空间 patch token，形状为 [B,N,D]。
        tokens = features["x_norm_patchtokens"]
        if isinstance(tokens, (list, tuple)):
            tokens = tokens[0]
        return tokens
