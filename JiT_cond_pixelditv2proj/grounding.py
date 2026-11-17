"""Frozen DINOv3 extraction from the current noisy state for patch AdaLN."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dinov3_backbone import load_dino_v3


class DINOv3GroundingEncoder(nn.Module):
    """从当前 noisy image 提取 DINOv3 patch token，不向 DINO 反传梯度。

    训练和采样统一编码 ``z_t``，不需要 ground-truth/reference/probe 图像。
    遵循 PixelDiT v2：对 noisy state 做 ``(z_t+1)/2``、ImageNet 标准化，
    并将输入尺寸对齐到 JiT patch 网格对应的 DINO 图像尺寸。噪声值可超出
    ``[-1,1]``，这里不做 clamp，也不把它当作已经位于图像取值范围内的数据。
    """

    def __init__(
        self,
        model_name: str = "dinov3_vitb16",
        weights_path: Optional[str] = None,
        repo: Optional[str] = None,
        target_patch_size: int = 16,
    ):
        """初始化冻结的 DINO 编码器和 JiT 目标 patch 网格配置。"""
        super().__init__()
        # JiT patch 边长: 标量 -> 16（B/16）或 32（B/32）。
        if not isinstance(target_patch_size, int) or target_patch_size <= 0:
            raise ValueError("target_patch_size must be a positive integer")
        self.target_patch_size = target_patch_size
        # 加载本地预训练 DINO；DINOv3-B patch feature: D_dino -> 768。
        self.model = load_dino_v3(model_name, weights_path=weights_path, repo=repo)
        # 冻结 encoder，训练梯度仅更新后续 projector 与 JiT。
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
        """即使外层模型进入训练模式，DINO 也始终保持 eval。"""
        # 某些 DINOv3 版本在 train 模式下会使用随机位置行为；作为冻结
        # encoder 时必须始终保持 eval，保证训练和采样的 token 稳定。
        super().train(False)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(self, noisy_images: torch.Tensor) -> torch.Tensor:
        """编码当前带噪图像；[B,3,H,W] -> [B,(H/P)*(W/P),D_dino]。

        JiT-B/16、256px 输入得到 ``[B,256,768]``。B/32、256px 时先把
        编码器输入缩放到 128px，得到与 JiT 一致的 ``[B,64,768]``。
        返回原始 DINO token；可训练时间 projector 位于 JiT 内，不在 no_grad 内。
        """
        # 验证通道/空间输入；noisy_images: [B,C,H,W] -> [B,3,256,256]。
        if noisy_images.ndim != 4 or noisy_images.shape[1] != 3:
            raise ValueError(f"expected noisy images [B,3,H,W], got {tuple(noisy_images.shape)}")
        # 按 JiT patch size 计算目标网格；grid_hw: (H/P,W/P) -> (16,16)。
        h, w = noisy_images.shape[-2:]
        if h <= 0 or w <= 0 or h % self.target_patch_size or w % self.target_patch_size:
            raise ValueError(
                f"input resolution {(h, w)} must be divisible by target_patch_size={self.target_patch_size}"
            )
        grid_hw = (h // self.target_patch_size, w // self.target_patch_size)
        # 将 noisy state 仿射映射后做 ImageNet 标准化；不裁剪噪声取值。
        # x: [B,3,H,W] -> [B,3,256,256]。
        x = noisy_images.float().add(1.0).mul(0.5)
        x = (x - self.image_mean.to(device=x.device)) / self.image_std.to(device=x.device)
        # Encoder 输入边长 = JiT 网格边长 * DINO patch16，确保 patch 一一对应。
        # encoder_hw: (grid_h*P_dino,grid_w*P_dino) -> (256,256)。
        encoder_hw = (grid_hw[0] * self.patch_size, grid_hw[1] * self.patch_size)
        if encoder_hw != (h, w):
            x = F.interpolate(x, size=encoder_hw, mode="bicubic", align_corners=False)
        # 冻结 DINO 编码当前 z_t；patch features: [B,N,D_dino] -> [B,256,768]。
        features = self.model.forward_features(x)
        # 去掉 CLS/register token，只保留空间 patch token，形状为 [B,N,D]。
        tokens = features["x_norm_patchtokens"]
        if isinstance(tokens, (list, tuple)):
            tokens = tokens[0]
        # 验证模型返回的 patch 数与目标网格匹配，防止 context/prefix token 混入。
        if tokens.ndim != 3 or tokens.shape[1] != grid_hw[0] * grid_hw[1] or tokens.shape[-1] != self.embed_dim:
            raise ValueError(
                f"DINO patch tokens must have shape [B,{grid_hw[0] * grid_hw[1]},{self.embed_dim}], "
                f"got {tuple(tokens.shape)}"
            )
        return tokens
