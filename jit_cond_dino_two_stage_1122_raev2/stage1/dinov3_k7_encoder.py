"""DINOv3-L k=7 encoder：聚合 DINOv3-L 的 7 层特征（层 12,14,16,18,20,22,24）。

与 RAEv2 完全一致的实现：
- 使用 get_intermediate_layers(n=[11,13,15,17,19,21,23], norm=True)
- 7 层平均后加上最后一层的空间平均（全局信息）
- Final norm 去掉 affine 参数
- z-score 归一化，使用 RAEv2 提供的 stats.pt

输出：[B, 1024, 16, 16]（DINOv3-L embed_dim=1024）
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPOSITORY_ROOT = PROJECT_ROOT.parent


def _resolve_dinov3_repo(repo: Optional[str] = None) -> Path:
    """查找 DINOv3 源码目录（包含 hubconf.py）。"""
    candidates = [
        Path(repo) if repo else None,
        Path(os.environ.get("DINOV3_REPO", "")),
        Path.home() / ".cache/torch/hub/facebookresearch_dinov3_main",
    ]
    for path in candidates:
        if path and path.is_dir() and (path / "hubconf.py").is_file():
            return path
    raise FileNotFoundError(
        "DINOv3 source tree not found. Set DINOV3_REPO to a directory containing hubconf.py; "
        f"tried {[str(p) for p in candidates if p]}"
    )


def _resolve_weights(weights_path: Optional[str] = None) -> Path:
    """查找 DINOv3-L 权重文件。"""
    if weights_path and Path(weights_path).is_file():
        return Path(weights_path)

    # 默认路径（与 RAEv2 一致）
    default_name = "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    candidates = [
        Path("/data0/pretrain_models") / default_name,
        Path(os.environ.get("DINOV3_CKPT_DIR", "")) / default_name,
        PROJECT_ROOT / "pretrained_models" / default_name,
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"DINOv3-L weights not found; tried {[str(p) for p in candidates]}")


def load_dinov3_vit_l16(weights_path: Optional[str] = None, repo: Optional[str] = None):
    """加载 DINOv3 ViT-L/16 模型。"""
    repo_dir = _resolve_dinov3_repo(repo)
    weights = _resolve_weights(weights_path)

    import torch.hub
    model = torch.hub.load(
        str(repo_dir),
        "dinov3_vitl16",
        source="local",
        trust_repo=True,
        skip_validation=True,
        weights=str(weights),
    )
    return model


class DINOv3Lk7Encoder(nn.Module):
    """DINOv3-L k=7 encoder，聚合 7 层特征，输出 [B, 1024, 16, 16]。

    与 RAEv2 DINOv3MultiLayerSimpleAddEncoder 完全一致：
    - Layer indices: [11, 13, 15, 17, 19, 21, 23]（0-indexed，对应 1-indexed [12,14,16,18,20,22,24]）
    - 聚合方式：7 层平均 + 最后一层空间平均
    - Final norm：elementwise_affine=False
    - z-score 归一化
    """

    LAYER_INDICES = [11, 13, 15, 17, 19, 21, 23]  # 0-indexed

    def __init__(
        self,
        input_size: int = 256,
        weights_path: Optional[str] = None,
        repo: Optional[str] = None,
        stat_path: Optional[str] = None,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.input_size = input_size
        self.eps = eps

        # 加载 DINOv3-L
        self.model = load_dinov3_vit_l16(weights_path, repo)
        self.model.eval()
        self.model.requires_grad_(False)

        self.embed_dim = self.model.embed_dim  # 1024
        self.patch_size = 16

        # 去掉 final norm 的 affine 参数
        self.model.norm = nn.LayerNorm(self.embed_dim, elementwise_affine=False)

        # ImageNet normalization
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

        # 加载 latent 统计量（z-score normalization）
        stat_path = self._resolve_stat_path(stat_path)
        stats = torch.load(stat_path, map_location="cpu", weights_only=True)
        mean = stats.get("mean")
        var = stats["var"]

        if mean is None:
            mean = torch.zeros(self.embed_dim)
        self.register_buffer("latent_mean", mean.view(1, -1, 1, 1), persistent=False)
        self.register_buffer("latent_var", var.view(1, -1, 1, 1), persistent=False)

    def _resolve_stat_path(self, override: Optional[str]) -> Path:
        """查找 DINOv3-L k=7 的统计文件。"""
        if override:
            return Path(override).expanduser()

        filename = "dinov3l_k7_imagenet1k_stats.pt"
        candidates = [
            PROJECT_ROOT / "stat" / filename,
            REPOSITORY_ROOT / "jit_cond_dino_two_stage_1122_raev2" / "stat" / filename,
        ]
        for path in candidates:
            if path.is_file():
                return path
        raise FileNotFoundError(
            f"DINOv3-L k=7 latent statistics missing: {filename}; set --dino_stat_path. "
            f"Download from https://huggingface.co/nyu-visionx/RAEv2-models/blob/main/stage1/imagenet/dinov3l-k7/stats.pt"
        )

    def train(self, mode: bool = True):
        """冻结 encoder，始终保持 eval 模式。"""
        super().train(False)
        return self

    @torch.no_grad()
    def encode(self, x01: torch.Tensor) -> torch.Tensor:
        """编码 [0,1] RGB 图像到归一化 latent [B, 1024, 16, 16]。

        Args:
            x01: [B, 3, H, W] 范围 [0, 1]

        Returns:
            [B, 1024, 16, 16] z-score 归一化后的 latent
        """
        x = x01.float()

        # Resize to 256x256
        if x.shape[-2:] != (self.input_size, self.input_size):
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bicubic", align_corners=False)

        # ImageNet normalize
        x = (x - self.img_mean) / self.img_std

        # 获取中间层特征（RAEv2 实现）
        outputs = self.model.get_intermediate_layers(
            x, n=self.LAYER_INDICES, reshape=False,
            return_class_token=False, norm=True
        )

        # 聚合：7 层平均 + 最后一层空间平均
        patch_tokens = torch.stack(outputs, dim=0).mean(dim=0)  # [B, N, D]
        final_mean = outputs[-1].mean(dim=1, keepdim=True)      # [B, 1, D]
        patch_tokens = patch_tokens + final_mean                 # 加全局信息

        # Reshape to [B, D, H, W]
        b, n, c = patch_tokens.shape
        h = w = int(n ** 0.5)
        assert h * w == n, f"patch count {n} is not a square grid"
        z = patch_tokens.transpose(1, 2).reshape(b, c, h, w)

        # z-score 归一化
        return (z - self.latent_mean) / torch.sqrt(self.latent_var + self.eps)
