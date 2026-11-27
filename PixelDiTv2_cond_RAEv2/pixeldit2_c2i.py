# Modified from https://github.com/MCG-NJU/PixNerd and https://github.com/LTH14/JiT

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import (
    ClassEmbedder,
    ModulatedDiTBlock,
    PatchFinalLayer,
    PatchTokenEmbedder,
    TimestepConditioner,
    precompute_freqs_cis_2d,
)


class PixelDiT2(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        hidden_size: int = 1280,
        num_heads: int = 16,
        depth: int = 32,
        patch_size: int = 16,
        num_classes: int = 1000,
        in_context_len: int = 32,
        in_context_start: int = 8,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.2,
        mlp_hidden_multiple: int = 0,
        single_silu_cond: bool = False,
        use_t_dino: bool = False,
    ):
        super().__init__()
        if patch_size != 16 or in_channels != 3:
            raise ValueError("This model expects RGB images and patch_size=16")
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.depth = depth
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.in_context_len = in_context_len
        self.in_context_start = in_context_start
        self.single_silu_cond = single_silu_cond
        self.use_t_dino = use_t_dino

        if not 0 <= in_context_start < depth:
            raise ValueError(
                f"in_context_start must be in [0, {depth - 1}], got {in_context_start}"
            )

        self.t_embedder = TimestepConditioner(hidden_size)
        self.t_dino_embedder = TimestepConditioner(hidden_size) if use_t_dino else None
        self.y_embedder = ClassEmbedder(num_classes + 1, hidden_size)
        self.s_embedder = PatchTokenEmbedder(in_channels * patch_size ** 2, hidden_size, bias=True)
        self.in_context_posemb = (
            nn.Parameter(torch.zeros(1, in_context_len, hidden_size))
            if in_context_len > 0
            else None
        )
        self.patch_blocks = nn.ModuleList(
            [
                ModulatedDiTBlock(
                    hidden_size,
                    num_heads,
                    attn_drop=attn_dropout if depth // 4 <= i < depth // 4 * 3 else 0.0,
                    proj_drop=proj_dropout if depth // 4 <= i < depth // 4 * 3 else 0.0,
                    mlp_hidden_multiple=mlp_hidden_multiple,
                )
                for i in range(depth)
            ]
        )
        self.final_layer = PatchFinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

        # Both clean DINOv3-L k=7 tokens and DDT samples use the same normalized latent space.
        self.grounding_proj = nn.Linear(1024, hidden_size)
        self._pos_cache: Dict[Tuple[int, int, int], torch.Tensor] = {}

    def initialize_weights(self):
        w = self.s_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.s_embedder.proj.bias, 0)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        if self.t_dino_embedder is not None:
            nn.init.normal_(self.t_dino_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_dino_embedder.mlp[2].weight, std=0.02)
        if self.in_context_posemb is not None:
            nn.init.normal_(self.in_context_posemb, std=0.02)
        for block in self.patch_blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)

    def fetch_pos(self, height: int, width: int, device, num_context_tokens: int = 0):
        key = (height, width, num_context_tokens)
        if key not in self._pos_cache:
            pos = precompute_freqs_cis_2d(self.hidden_size // self.num_heads, height, width)
            if num_context_tokens > 0:
                identity = torch.ones(num_context_tokens, pos.shape[-1], dtype=pos.dtype)
                pos = torch.cat([identity, pos], dim=0)
            self._pos_cache[key] = pos
        return self._pos_cache[key].to(device)

    def _grounded_condition(self, grounding_tokens, t_emb, y_emb, t_dino=None):
        grounding = self.grounding_proj(grounding_tokens)
        c = grounding + t_emb.unsqueeze(1) + y_emb
        if self.t_dino_embedder is not None:
            if t_dino is None:
                raise ValueError("t_dino is required when DINO time conditioning is enabled")
            c = c + self.t_dino_embedder(t_dino).unsqueeze(1)
        return c if self.single_silu_cond else F.silu(c)

    def _null_condition(self, t_emb, y_emb, num_patches):
        c = t_emb.unsqueeze(1) + y_emb
        if not self.single_silu_cond:
            c = F.silu(c)
        return c.expand(-1, num_patches, -1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        grounding_tokens: Optional[torch.Tensor] = None,
        skip_grounding: bool = False,
        repa_align_layer: Optional[int] = None,
        t_dino: Optional[torch.Tensor] = None,
    ):
        """Optionally return patch features from the specified one-based REPA layer."""
        batch, channels, height, width = x.shape
        if (channels, height, width) != (3, 256, 256):
            raise ValueError(f"Expected ImageNet RGB input [B,3,256,256], got {tuple(x.shape)}")
        if repa_align_layer is not None and not 1 <= repa_align_layer <= self.depth:
            raise ValueError(f"repa_align_layer must be between 1 and {self.depth}")
        t = t.view(batch)
        hs, ws = height // self.patch_size, width // self.patch_size
        num_patches = hs * ws

        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y).view(batch, 1, self.hidden_size)

        if skip_grounding:
            c = self._null_condition(t_emb, y_emb, num_patches)
        else:
            expected = (batch, num_patches, self.grounding_proj.in_features)
            if grounding_tokens is None or grounding_tokens.shape != expected:
                actual = None if grounding_tokens is None else tuple(grounding_tokens.shape)
                raise ValueError(f"Expected grounding_tokens of shape {expected}, got {actual}")
            c = self._grounded_condition(grounding_tokens, t_emb, y_emb, t_dino)

        s = self.s_embedder(
            F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)
        )

        use_context = self.in_context_len > 0
        pos = self.fetch_pos(hs, ws, x.device)
        pos_with_context = self.fetch_pos(hs, ws, x.device, self.in_context_len) if use_context else None
        if use_context:
            c_context = c.mean(dim=1, keepdim=True).expand(-1, self.in_context_len, -1)
            c_extended = torch.cat([c_context, c], dim=1)

        context_inserted = False
        for i, block in enumerate(self.patch_blocks):
            if use_context and not context_inserted and i == self.in_context_start:
                context_tokens = y_emb.expand(-1, self.in_context_len, -1)
                s = torch.cat([context_tokens + self.in_context_posemb.to(s.dtype), s], dim=1)
                context_inserted = True
            s = block(
                s,
                c_extended if context_inserted else c,
                pos_with_context if context_inserted else pos,
            )
            if i + 1 == repa_align_layer:
                # For H/16, block 8 completes before context is added at block 9.
                repa_features = s[:, self.in_context_len:, :] if context_inserted else s
        if context_inserted:
            s = s[:, self.in_context_len:, :]

        out = self.final_layer(s, c)
        out = out.view(batch, num_patches, self.patch_size ** 2, self.out_channels)
        out = out.permute(0, 3, 2, 1).reshape(batch, self.out_channels * self.patch_size ** 2, num_patches)
        out = F.fold(out, (height, width), kernel_size=self.patch_size, stride=self.patch_size)

        if repa_align_layer is not None:
            return out, repa_features
        return out
