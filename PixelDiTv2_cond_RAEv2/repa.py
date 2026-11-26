"""Frozen REPA teacher and alignment loss from PixelDiT2 grounded_diffusion."""

import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize


REPA_ENCODERS = {
    "dinov2_vitb14": ("vit_base_patch14_dinov2.lvd142m", 14, 768),
    "dinov2_vitl14": ("vit_large_patch14_dinov2.lvd142m", 14, 1024),
}

# Offset used by DINOv2 positional-embedding interpolation.
_DINOV2_INTERPOLATE_OFFSET = 0.1


def _dinov2_pos_embed(self, x):
    """Apply positional embeddings with DINOv2 bicubic resampling."""
    batch, height, width, channels = x.shape
    grid = self.patch_embed.grid_size[0]
    n_prefix = self.num_prefix_tokens
    pos = self.pos_embed.float()
    prefix_pos, patch_pos = pos[:, :n_prefix], pos[:, n_prefix:]
    patch_pos = patch_pos.reshape(1, grid, grid, channels).permute(0, 3, 1, 2)
    if (height, width) != (grid, grid):
        scale = (
            float(width + _DINOV2_INTERPOLATE_OFFSET) / grid,
            float(height + _DINOV2_INTERPOLATE_OFFSET) / grid,
        )
        patch_pos = F.interpolate(patch_pos, scale_factor=scale, mode="bicubic", antialias=False)
        if tuple(patch_pos.shape[-2:]) != (height, width):
            raise RuntimeError(
                f"positional embedding resampled to {tuple(patch_pos.shape[-2:])}, "
                f"expected {(height, width)}"
            )
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, height * width, channels)
    pos = torch.cat([prefix_pos, patch_pos], dim=1).to(x.dtype)
    x = x.view(batch, -1, channels)
    prefix = [
        token.expand(batch, -1, -1)
        for token in (self.cls_token, getattr(self, "reg_token", None))
        if token is not None
    ]
    x = torch.cat(prefix + [x], dim=1)
    return self.pos_drop(x + pos)


class REPATeacher(nn.Module):
    """Frozen DINOv2 encoder for REPA alignment targets."""

    def __init__(
        self,
        weights_path=None,
        model_name: str = "dinov2_vitb14",
        bf16_weights: bool = True,
        dinov2_pos_embed: bool = True,
    ):
        import timm

        super().__init__()
        timm_name, self.patch_size, self.feature_dim = REPA_ENCODERS[model_name]
        self.encoder = timm.create_model(
            timm_name,
            pretrained=True,
            dynamic_img_size=True,
            pretrained_cfg_overlay={"file": str(weights_path)} if weights_path else None,
        )
        self.encoder.requires_grad_(False)
        if bf16_weights:
            self.encoder.to(torch.bfloat16)
        if dinov2_pos_embed:
            if getattr(self.encoder, "no_embed_class", False) or not getattr(
                self.encoder, "dynamic_img_size", False
            ):
                raise ValueError(
                    "dinov2_pos_embed needs a timm ViT built with dynamic_img_size=True "
                    "whose positional embedding covers the class token"
                )
            self.encoder._pos_embed = types.MethodType(_dinov2_pos_embed, self.encoder)
        self.bf16_weights = bf16_weights
        self.dinov2_pos_embed = dinov2_pos_embed

        self.eval()

    def train(self, mode: bool = True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, images, denoiser_patch_size: int):
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(images)
        height, width = x.shape[-2:]
        x = F.interpolate(
            x,
            (
                int(self.patch_size * height / denoiser_patch_size),
                int(self.patch_size * width / denoiser_patch_size),
            ),
            mode="bicubic",
        )
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16):
            features = self.encoder.forward_features(x)
        prefix = int(getattr(self.encoder, "num_prefix_tokens", 0))
        return features[:, prefix:, :].to(torch.bfloat16)


def build_repa_projection(hidden_size: int = 1280, output_dim: int = 768):
    return nn.Sequential(
        nn.Linear(hidden_size, hidden_size),
        nn.SiLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.SiLU(),
        nn.Linear(hidden_size, output_dim),
    )


def _match_token_grid(src, target_len):
    if src.shape[1] == target_len:
        return src
    batch, length, channels = src.shape
    side, target_side = int(length ** 0.5), int(target_len ** 0.5)
    spatial = src.reshape(batch, side, side, channels).permute(0, 3, 1, 2)
    if target_len < length:
        spatial = F.adaptive_avg_pool2d(spatial, (target_side, target_side))
    else:
        spatial = F.interpolate(
            spatial, size=(target_side, target_side), mode="bilinear", align_corners=False
        )
    return spatial.permute(0, 2, 3, 1).reshape(batch, target_len, channels)


def repa_cosine_loss(projected_features, teacher_features):
    src = _match_token_grid(projected_features, teacher_features.shape[1])
    return (1.0 - F.cosine_similarity(src, teacher_features, dim=-1)).mean()
