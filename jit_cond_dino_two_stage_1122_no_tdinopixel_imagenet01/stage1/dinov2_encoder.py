"""统一的 DINO latent 编码器（join-dino 的 grounding / dino DiT 目标空间）。

支持两种冻结编码器，两者最后的归一化层都**去掉 affine**，与 RAE / RAEv2 / LF
对齐：

* ``dinov2b``：DINOv2-reg-B/14（facebook/dinov2-with-registers-base），bicubic
  224² 输入，丢 1 CLS + 4 register，最后 LayerNorm 去 affine。
* ``dinov3b``：DINOv3-B/16（dinov3_vitb16，SHA 73cec8be），256² 输入即 16×16 网格，
  取 ``forward_features['x_norm_patchtokens']``，最后 norm 去 affine。

两者归一化统一为 ``(z − mean) / sqrt(var+eps)``，**不再乘 0.485**（LatentForcing 独有的
match_pixel_norm 已去掉，与 RAE v1/v2 口径一致）。两者都输出 ``[B, D, 16, 16]``（D=768）。
统计文件放在本项目的 ``stat/`` 目录，key 为 ``mean`` / ``var``（``mean`` 为 None 时不减均值）。
"""

from __future__ import annotations

import os
from math import sqrt
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from resource_paths import PROJECT_ROOT, REPOSITORY_ROOT, pretrain_models_root

PRETRAIN_DIR = str(pretrain_models_root())

# 每种编码器的默认配置：权重名、输入分辨率、统计文件。归一化统一不乘额外系数。
ENCODER_PRESETS = {
    "dinov2b": {
        "kind": "dinov2",
        "weights": (os.path.join(PRETRAIN_DIR, "dinov2-with-registers-base")
                    if os.path.isdir(os.path.join(PRETRAIN_DIR, "dinov2-with-registers-base"))
                    else "facebook/dinov2-with-registers-base"),
        "input_size": 224,
        "stat": "dinov2b_rae_imagenet1k_stat.pt",
    },
    "dinov3b": {
        "kind": "dinov3",
        "weights": "dinov3_vitb16",
        "input_size": 256,
        "stat": "dinov3b_rae_imagenet1k_stats.pt",
    },
}



def latent_stat_path(encoder_type, override=None):
    if override:
        return Path(override).expanduser()
    filename = ENCODER_PRESETS[encoder_type]["stat"]
    candidates = (
        PROJECT_ROOT / "stat" / filename,
        REPOSITORY_ROOT / "jit_cond_dino_two_stage_1122_no_tdinopixel" / "stat" / filename,
        REPOSITORY_ROOT / "JiT_cond_join_dino" / "stat" / filename,
        REPOSITORY_ROOT / "jit_cond_predict_independent_dino" / "stat" / filename,
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"DINO latent statistics missing: {filename}; set --dino_stat_path")


class _DINOv2Backbone(nn.Module):
    """冻结 DINOv2-reg-B/14，最后 LayerNorm 去 affine，返回 patch token [B,N,D]。"""

    def __init__(self, weights: str):
        super().__init__()
        from transformers import Dinov2WithRegistersModel
        self.model = Dinov2WithRegistersModel.from_pretrained(weights, local_files_only=True)
        self.model.requires_grad_(False)
        self.model.eval()
        self.model.layernorm.elementwise_affine = False
        self.model.layernorm.weight = None
        self.model.layernorm.bias = None
        self.hidden_size = self.model.config.hidden_size

    def forward(self, x):
        out = self.model(x, output_hidden_states=True)
        return out.last_hidden_state[:, 5:]      # 丢 1 CLS + 4 register


class _DINOv3Backbone(nn.Module):
    """冻结 DINOv3-B/16，最后 norm 去 affine，返回 patch token [B,N,D]。"""

    def __init__(self, weights: str, weights_path: Optional[str], repo: Optional[str]):
        super().__init__()
        from .dinov3_backbone import load_dino_v3
        self.model = load_dino_v3(weights, weights_path=weights_path, repo=repo)
        self.model.requires_grad_(False)
        self.model.eval()
        self.hidden_size = int(getattr(self.model, "embed_dim", getattr(self.model, "num_features", 768)))
        # 去掉最后归一化层的 affine（与 RAEv2 一致）。
        self.model.norm = nn.LayerNorm(self.hidden_size, elementwise_affine=False)

    def forward(self, x):
        feats = self.model.forward_features(x)
        tokens = feats["x_norm_patchtokens"]
        if isinstance(tokens, (list, tuple)):
            tokens = tokens[0]
        return tokens


class DINOLatentEncoder(nn.Module):
    """``encode(x01) -> [B, D, 16, 16]``。x01 为 [0,1] RGB 图像。"""

    def __init__(
        self,
        encoder_type: str = "dinov2b",
        weights: Optional[str] = None,
        weights_path: Optional[str] = None,
        repo: Optional[str] = None,
        stat_path: Optional[str] = None,
        input_size: Optional[int] = None,
        eps: float = 1e-5,
    ):
        super().__init__()
        if encoder_type not in ENCODER_PRESETS:
            raise ValueError(f"unknown encoder_type {encoder_type!r}; choices={list(ENCODER_PRESETS)}")
        cfg = ENCODER_PRESETS[encoder_type]
        self.encoder_type = encoder_type
        self.kind = cfg["kind"]
        self.input_size = int(input_size or cfg["input_size"])
        self.eps = float(eps)

        w = weights or cfg["weights"]
        if self.kind == "dinov2":
            self.backbone = _DINOv2Backbone(w)
        else:
            self.backbone = _DINOv3Backbone(w, weights_path, repo)
        self.latent_dim = self.backbone.hidden_size

        self.register_buffer("encoder_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("encoder_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

        stat_path = latent_stat_path(encoder_type, stat_path)
        stats = torch.load(stat_path, map_location="cpu", weights_only=True)
        mean, var = stats.get("mean", None), stats.get("var", None)
        self.register_buffer("latent_mean", mean if mean is not None else torch.zeros(()), persistent=False)
        self.register_buffer("latent_var", var if var is not None else torch.ones(()), persistent=False)

    def train(self, mode: bool = True):
        # 冻结 teacher：始终 eval，不随外层 model.train() 改变。
        super().train(False)
        return self

    @torch.no_grad()
    def encode(self, x01: torch.Tensor) -> torch.Tensor:
        x = x01.float()
        if x.shape[-2:] != (self.input_size, self.input_size):
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bicubic", align_corners=False)
            # bicubic 在强对比边缘可能超出 [0,1]，在按通道做 ImageNet 归一化之前限制 RGB 范围。
            x = x.clamp(0.0, 1.0)
        x = (x - self.encoder_mean) / self.encoder_std
        z = self.backbone(x)                             # [B,N,D]
        b, n, c = z.shape
        h = w = int(sqrt(n))
        assert h * w == n, f"patch count {n} is not a square grid"
        z = z.transpose(1, 2).reshape(b, c, h, w)
        return (z - self.latent_mean) / torch.sqrt(self.latent_var + self.eps)


# 向后兼容旧引用名。
DINOv2LatentEncoder = DINOLatentEncoder
