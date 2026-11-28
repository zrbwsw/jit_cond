"""DINO DiT：独立去噪 Latent Forcing 的 dino path。

结构对齐 LatentForcing/model_cot.py 中 dino 分支用到的部件：
* 输入：``dino_embedder = BottleneckPatchEmbed(G, patch=1, 768 -> 128 -> C)``（两层 1×1 卷积）；
* 固定二维 sin-cos 位置编码；类别 in-context 前缀（32 个，第 ``in_context_start`` 层前插入）；
* ``JiTBlock`` 堆叠，全局 AdaLN 条件 ``c = t_dino_emb + y_emb``；
* ``FinalLayer`` x-prediction，输出每个位置 768 维，unpatchify(p=1) 回 ``[B,768,16,16]``。

与 LF 的区别：LF 把 pixel/dino 两路 embedding 逐元素相加后共享 trunk；这里 dino
path 是独立网络（JiT-B 同尺寸：12 层、768 宽、12 头），两路的交互改为在 JiT 侧
``c = t + y + W_g·grounding``。因此不再有 ``t_pixel_embedder`` 和 pixel 输入。
"""

import torch
import torch.nn as nn

from .lf_modules import (
    BottleneckPatchEmbed,
    FinalLayer,
    JiTBlock,
    LabelEmbedder,
    TimestepEmbedder,
)
from .model_util import VisionRotaryEmbeddingFast, get_2d_sincos_pos_embed


class DinoDiT(nn.Module):
    def __init__(
        self,
        grid_size=16,
        dino_in_channels=768,
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        num_classes=1000,
        bottleneck_dim_dino=128,
        in_context_len=32,
        in_context_start=4,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.dino_in_channels = dino_in_channels
        self.hidden_size = hidden_size
        self.in_context_len = in_context_len
        self.in_context_start = in_context_start
        self.num_classes = num_classes

        self.t_dino_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)
        # 与 LF 一致：dino path 经过 bottleneck（patch=1 的两层 1×1 卷积）。
        self.dino_embedder = BottleneckPatchEmbed(grid_size, 1, dino_in_channels, bottleneck_dim_dino,
                                                  hidden_size, bias=True)

        num_patches = self.dino_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        if self.in_context_len > 0:
            self.in_context_posemb = nn.Parameter(torch.zeros(1, self.in_context_len, hidden_size), requires_grad=True)
            torch.nn.init.normal_(self.in_context_posemb, std=.02)

        half_head_dim = hidden_size // num_heads // 2
        self.feat_rope = VisionRotaryEmbeddingFast(dim=half_head_dim, pt_seq_len=grid_size, num_cls_token=0)
        self.feat_rope_incontext = VisionRotaryEmbeddingFast(dim=half_head_dim, pt_seq_len=grid_size,
                                                             num_cls_token=self.in_context_len)

        # 与 LF / JiT 相同：dropout 只作用在中间一半的层。
        self.blocks = nn.ModuleList([
            JiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                     attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                     proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0)
            for i in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, dino_in_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # 与 LF JiTCoT.initialize_weights 中 dino 分支相关的部分一致。
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w1 = self.dino_embedder.proj1.weight.data
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        w2 = self.dino_embedder.proj2.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        nn.init.constant_(self.dino_embedder.proj2.bias, 0)

        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_dino_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_dino_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, z_dino, t_d, y):
        """z_dino: [B,768,16,16] 带噪 latent；t_d: [B]；y: [B]。返回 x-pred，形状同 z_dino。"""
        c = self.t_dino_embedder(t_d) + self.y_embedder(y)
        y_emb = self.y_embedder(y)

        x = self.dino_embedder(z_dino)
        x = x + self.pos_embed
        for i, block in enumerate(self.blocks):
            if self.in_context_len > 0 and i == self.in_context_start:
                in_context_tokens = y_emb.unsqueeze(1).repeat(1, self.in_context_len, 1)
                in_context_tokens = in_context_tokens + self.in_context_posemb
                x = torch.cat([in_context_tokens, x], dim=1)
            x = block(x, c, self.feat_rope if i < self.in_context_start else self.feat_rope_incontext)
        x = x[:, self.in_context_len:]

        x = self.final_layer(x, c)                                   # [B,N,768]
        b, n, d = x.shape
        return x.transpose(1, 2).reshape(b, d, self.grid_size, self.grid_size)


def DinoDiT_B(**kwargs):
    """JiT-B 同尺寸：12 层、hidden 768、12 头、第 4 层前插入类别前缀。"""
    return DinoDiT(depth=12, hidden_size=768, num_heads=12, in_context_len=32, in_context_start=4, **kwargs)


DinoDiT_models = {"DinoDiT-B": DinoDiT_B}
