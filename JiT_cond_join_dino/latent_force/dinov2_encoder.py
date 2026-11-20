"""与 Latent Forcing 完全一致的 DINOv2 latent 编码器（训练目标 / grounding 空间）。

逻辑逐行对应 LatentForcing/dinov2_hf.py 上游 HEAD 的 ``Dinov2withNorm`` 与
``RAE.encode``：

    [0,1] RGB -> bicubic 224² -> ImageNet mean/std
    -> facebook/dinov2-with-registers-base（最后 LayerNorm 去掉 affine）
    -> 丢 1 CLS + 4 register -> [B,768,16,16]
    -> (z - mean) / sqrt(var + 1e-5)       # RAE ImageNet-1k stat.pt，mean=None
    -> × 0.485                             # match_pixel_norm

改动只有三处，都不改变数值：
* 只在本地加载权重（``local_files_only``），不联网；
* 统计文件默认指向本项目随附的 ``dinov2_wreg_base_imagenet1k_stat.pt``；
* 去掉 RAE 里未使用的 noise_tau 分支和 ``@torch.compile``。

``facebook/dinov2-with-registers-base`` 与 lfjit-l-200.pth 的 ``model`` 字典中
``dino.*`` 逐张量完全相同（222 个张量，max|diff|=0）。
"""

from __future__ import annotations

import os
from math import sqrt
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_STAT_PATH = os.path.join(_HERE, "dinov2_wreg_base_imagenet1k_stat.pt")
DEFAULT_DINOV2 = "facebook/dinov2-with-registers-base"
MATCH_PIXEL_NORM = 0.485


class Dinov2withNorm(nn.Module):
    """冻结的 DINOv2-reg-B/14；``normalize=True`` 时最后 LN 不带 gamma/beta。"""

    def __init__(self, dinov2_path: str = DEFAULT_DINOV2, normalize: bool = True):
        super().__init__()
        from transformers import Dinov2WithRegistersModel
        self.encoder = Dinov2WithRegistersModel.from_pretrained(dinov2_path, local_files_only=True)
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        if normalize:
            self.encoder.layernorm.elementwise_affine = False
            self.encoder.layernorm.weight = None
            self.encoder.layernorm.bias = None
        self.patch_size = self.encoder.config.patch_size
        self.hidden_size = self.encoder.config.hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x, output_hidden_states=True)
        unused_token_num = 5  # 1 CLS + 4 register tokens
        return x.last_hidden_state[:, unused_token_num:]


class DINOv2LatentEncoder(nn.Module):
    """``encode(x01) -> [B,768,16,16]``，即 LF 的 x_dino 空间。"""

    def __init__(
        self,
        dinov2_path: str = DEFAULT_DINOV2,
        normalization_stat_path: Optional[str] = DEFAULT_STAT_PATH,
        encoder_input_size: int = 224,
        match_pixel_norm: float = MATCH_PIXEL_NORM,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.encoder = Dinov2withNorm(dinov2_path, normalize=True)
        self.encoder_input_size = encoder_input_size
        self.latent_dim = self.encoder.hidden_size
        self.grid = encoder_input_size // self.encoder.patch_size
        self.register_buffer("encoder_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("encoder_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

        stats = torch.load(normalization_stat_path, map_location="cpu", weights_only=True)
        mean, var = stats.get("mean", None), stats.get("var", None)
        self.register_buffer("latent_mean", mean if mean is not None else torch.zeros(()), persistent=False)
        self.register_buffer("latent_var", var if var is not None else torch.ones(()), persistent=False)
        self.match_pixel_norm = float(match_pixel_norm)
        self.eps = float(eps)

    def train(self, mode: bool = True):
        # 冻结 teacher：始终保持 eval，避免外层 model.train() 改变其行为。
        super().train(False)
        return self

    @torch.no_grad()
    def encode(self, x01: torch.Tensor) -> torch.Tensor:
        x = x01.float()
        if x.shape[-2:] != (self.encoder_input_size, self.encoder_input_size):
            x = F.interpolate(x, size=(self.encoder_input_size, self.encoder_input_size),
                              mode="bicubic", align_corners=False)
        x = (x - self.encoder_mean) / self.encoder_std
        z = self.encoder(x)
        b, n, c = z.shape
        h = w = int(sqrt(n))
        z = z.transpose(1, 2).reshape(b, c, h, w)
        z = (z - self.latent_mean) / torch.sqrt(self.latent_var + self.eps)
        return z * self.match_pixel_norm
