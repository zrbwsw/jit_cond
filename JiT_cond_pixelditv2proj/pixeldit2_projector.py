"""PixelDiT2 timestep projector and its required attention components.

Adapted from PixelDiT2 pixdit_core/{grounding,modules}.py at commit
8f83fa88a7b9dab575e5a45dafe3bb3c308685e0. Copyright (c) NVIDIA Corporation.
See third_party/pixeldit2/LICENSE and NOTICE for the upstream terms/attribution.
The projector's operations, state-dict names and initialization are preserved.
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.nn.functional import scaled_dot_product_attention


def apply_adaln(x, shift, scale):
    # x: [B,N,C]；shift/scale: [B,1,C]，广播作用到每个 token
    return x * (1 + scale) + shift


class TimestepConditioner(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10):
        # t: [B] 标量时间 → [B, dim]，前半 cos、后半 sin 的正弦频率嵌入
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[..., None].float() * freqs[None, ...]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        # t: [B] → [B, hidden_size]
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        mlp_dtype = next(self.mlp.parameters()).dtype
        if t_freq.dtype != mlp_dtype:
            t_freq = t_freq.to(mlp_dtype)
        return self.mlp(t_freq)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # hidden_states: [..., C]，在末维做 RMS 归一化；weight: [C] 可学习增益
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.0, hidden_multiple=0):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_multiple = int(hidden_multiple)
        if hidden_multiple > 0:
            hidden_dim = ((hidden_dim + hidden_multiple - 1) // hidden_multiple) * hidden_multiple
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        # x: [B,N,dim] → [B,N,dim]；SwiGLU 结构：silu(w1(x)) * w3(x) 再投影回 dim
        hidden = torch.nn.functional.silu(self.w1(x)) * self.w3(x)
        return self.w2(self.ffn_dropout(hidden))


def precompute_freqs_cis_2d(dim, height, width, theta=10000.0, scale=16.0):
    # 预生成 2D RoPE 频率表；返回 [H*W, dim//2] 复数张量（dim 为 head_dim）
    x_pos = torch.linspace(0, scale, width)
    y_pos = torch.linspace(0, scale, height)
    y_pos, x_pos = torch.meshgrid(y_pos, x_pos, indexing="ij")
    y_pos = y_pos.reshape(-1)
    x_pos = x_pos.reshape(-1)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    x_freqs = torch.outer(x_pos, freqs).float()
    y_freqs = torch.outer(y_pos, freqs).float()
    x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
    y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
    freqs_cis = torch.cat([x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1)
    return freqs_cis.reshape(height * width, -1)


def apply_rotary_emb(xq, xk, freqs_cis):
    # xq/xk: [B,N,heads,head_dim]；freqs_cis: [N,head_dim//2] 复数；返回与输入同形
    freqs_cis = freqs_cis[None, :, None, :]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RotaryAttention(nn.Module):
    def __init__(
        self, dim, num_heads=8, qkv_bias=False, qk_norm=True,
        attn_drop=0.0, proj_drop=0.0, norm_layer=RMSNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, pos, mask):
        # x: [B,N,C]；pos: [N,head_dim//2]（freqs_cis）；mask: 注意力掩码或 None
        batch, length, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch, length, 3, self.num_heads, channels // self.num_heads,
        ).permute(2, 0, 1, 3, 4)  # → [3, B, N, heads, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = apply_rotary_emb(q, k, freqs_cis=pos)  # 注入 2D 位置，形状不变
        q = q.view(batch, -1, self.num_heads, channels // self.num_heads).transpose(1, 2)
        k = k.view(batch, -1, self.num_heads, channels // self.num_heads).transpose(1, 2).contiguous()
        v = v.view(batch, -1, self.num_heads, channels // self.num_heads).transpose(1, 2).contiguous()
        # q/k/v: [B, heads, N, head_dim]
        x = scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        x = x.transpose(1, 2).reshape(batch, length, channels)  # → [B, N, C]
        return self.proj_drop(self.proj(x))


class ModulatedDiTBlock(nn.Module):
    def __init__(
        self, hidden_size, num_heads, mlp_ratio=4.0,
        attn_drop=0.0, proj_drop=0.0, mlp_hidden_multiple=0,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = RotaryAttention(
            hidden_size, num_heads=num_heads, qkv_bias=False,
            attn_drop=attn_drop, proj_drop=proj_drop,
        )
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = FeedForward(
            hidden_size, int(hidden_size * mlp_ratio),
            drop=proj_drop, hidden_multiple=mlp_hidden_multiple,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x, c, pos, mask=None):
        # x: [B,N,C]；c: [B,1,C] 时间条件向量；pos: [N,head_dim//2]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)  # 各 [B,1,C]，adaLN 六路调制参数
        )
        x = x + gate_msa * self.attn(apply_adaln(self.norm1(x), shift_msa, scale_msa), pos, mask=mask)
        x = x + gate_mlp * self.mlp(apply_adaln(self.norm2(x), shift_mlp, scale_mlp))
        return x


class TimestepProjection(nn.Module):
    """用时间条件 Transformer 翻译 DINO patch token，并映射到 JiT hidden size。

    ``tokens:[B,N,in_dim]``、原始扩散时间 ``t:[B]``、网格 ``(grid_h,grid_w)``
    输入后返回 ``[B,N,out_dim]``。JiT-B/16 默认 768 -> 768，1 个 block、12 heads。
    输入与输出 Linear 负责维度映射；中间 block 使用独立时间嵌入、AdaLN、二维 RoPE。
    """
    def __init__(
        self, in_dim, out_dim, hidden_size,
        num_blocks=1, num_heads=16, mlp_ratio=4.0,
    ):
        """构造 input/output Linear 和时间调制 block，保留 PixelDiT v2 初始化。"""
        super().__init__()
        # 非空 projector 才具备时间条件处理路径；blocks: 标量 -> 1。
        if not isinstance(num_blocks, int) or num_blocks < 1:
            raise ValueError("TimestepProjection num_blocks must be a positive integer")
        # 检查投影维度和 head 数，避免除零或无法拆分 head。
        if any(not isinstance(size, int) or size < 1 for size in (in_dim, out_dim, hidden_size, num_heads)):
            raise ValueError("TimestepProjection dimensions and num_heads must be positive integers")
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"TimestepProjection hidden_size={hidden_size} must be divisible by "
                f"num_heads={num_heads}"
            )
        self.head_dim = hidden_size // num_heads
        # 二维复数 RoPE 分别给 x/y 分配成对 feature；head_dim: 768/12 -> 64。
        if self.head_dim % 4:
            raise ValueError("TimestepProjection head_dim must be divisible by 4 for 2D RoPE")
        # 输入维度映射；[B,N,in_dim] -> [B,N,hidden_size]，B/16: [B,256,768]。
        self.input_proj = nn.Linear(in_dim, hidden_size, bias=True)
        # 独立编码原始 t，不转换为 1-t，也不乘 1000；[B] -> [B,768]。
        self.t_embedder = TimestepConditioner(hidden_size)
        # 时间调制 Transformer；[B,N,hidden_size] -> [B,256,768]。
        self.blocks = nn.ModuleList([
            ModulatedDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(num_blocks)
        ])
        # 输出 hidden 映射；[B,N,hidden_size] -> [B,N,out_dim]，B/16: [B,256,768]。
        self.output_proj = nn.Linear(hidden_size, out_dim, bias=True)
        # 仅清零 block 的 AdaLN；初始 block 为 identity，input/output Linear 保持默认初始化。
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        self._pos_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def fetch_pos(self, height, width, device):
        """缓存给定二维网格的复数 RoPE；[grid_h*grid_w,head_dim//2] -> [256,32]。"""
        key = (height, width)
        if key not in self._pos_cache:
            self._pos_cache[key] = precompute_freqs_cis_2d(self.head_dim, height, width)
        return self._pos_cache[key].to(device)

    def forward(self, tokens, t, grid_hw):
        """将当前 noisy DINO patch token 与同一状态的 t 映射为 patch 条件。"""
        # 验证 token feature 维；[B,N,in_dim] -> [B,256,768]。
        if tokens.ndim != 3 or tokens.shape[-1] != self.input_proj.in_features:
            raise ValueError(f"expected tokens [B,N,{self.input_proj.in_features}], got {tuple(tokens.shape)}")
        # 验证每个样本只有一个时间值；t: [B]。
        if t.ndim != 1 or t.shape[0] != tokens.shape[0]:
            raise ValueError("projector timestep must have shape [B] matching tokens")
        # 验证空间网格和 patch 数一致；grid_hw: (16,16)，N: 256。
        if len(grid_hw) != 2 or any(size <= 0 for size in grid_hw) or grid_hw[0] * grid_hw[1] != tokens.shape[1]:
            raise ValueError(f"projector grid {grid_hw} must match token count {tokens.shape[1]}")
        # 输入维度映射；x: [B,N,hidden_size] -> [B,256,768]。
        x = self.input_proj(tokens)
        # 时间条件扩展 token 维，给所有 patch 广播；c: [B,1,hidden_size] -> [B,1,768]。
        c = self.t_embedder(t).unsqueeze(1).to(dtype=x.dtype)
        # 二维 RoPE 表；pos: [N,head_dim//2] -> [256,32]（复数）。
        pos = self.fetch_pos(grid_hw[0], grid_hw[1], x.device)
        # 依次执行时间 AdaLN attention/FFN 残差块；x: [B,N,hidden_size] -> [B,256,768]。
        for block in self.blocks:
            x = block(x, c, pos)
        # 最终维度映射后直接作为 grounding 条件；[B,N,out_dim] -> [B,256,768]。
        return self.output_proj(x)
