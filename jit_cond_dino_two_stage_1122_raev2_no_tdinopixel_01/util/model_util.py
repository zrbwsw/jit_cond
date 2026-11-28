# --------------------------------------------------------
# References:
# Lightning-DiT: https://github.com/hustvl/LightningDiT
# --------------------------------------------------------

from math import pi

import torch
from torch import nn
import numpy as np

from einops import rearrange, repeat


def broadcat(tensors, dim = -1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim = dim)


def rotate_half(x):
    """把最后一维按相邻二元组做 90° 旋转，供 RoPE 构造正交分量。

    输入 ``x:[B, num_heads, N, D]``，JiT-B attention 中为
    ``[B,12,256/288,64]``；输出形状保持不变。
    """
    # 将最后一维按两个 feature 一组重排；[B,12,N,64] -> [B,12,N,32,2]。
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    # 拆出每个二元组的两个分量；[B,12,N,32,2] -> 两个 [B,12,N,32]。
    x1, x2 = x.unbind(dim = -1)
    # 形成 (-x2, x1)，实现每个二元组的 90° 旋转；[B,12,N,32] -> [B,12,N,32,2]。
    x = torch.stack((-x2, x1), dim = -1)
    # 合并二元组维度；[B,12,N,32,2] -> [B,12,N,64]。
    return rearrange(x, '... d r -> ... (d r)')


class VisionRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len,
        ft_seq_len=None,
        custom_freqs = None,
        freqs_for = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        if ft_seq_len is None: ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs_h = torch.einsum('..., f -> ... f', t, freqs)
        freqs_h = repeat(freqs_h, '... n -> ... (n r)', r = 2)

        freqs_w = torch.einsum('..., f -> ... f', t, freqs)
        freqs_w = repeat(freqs_w, '... n -> ... (n r)', r = 2)

        freqs = broadcat((freqs_h[:, None, :], freqs_w[None, :, :]), dim = -1)

        self.register_buffer("freqs_cos", freqs.cos())
        self.register_buffer("freqs_sin", freqs.sin())

    def forward(self, t, start_index = 0):
        rot_dim = self.freqs_cos.shape[-1]
        end_index = start_index + rot_dim
        assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
        t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
        t = (t * self.freqs_cos) + (rotate_half(t) * self.freqs_sin)
        return torch.cat((t_left, t, t_right), dim = -1)


class VisionRotaryEmbeddingFast(nn.Module):
    """为视觉 token 的 query/key 注入二维旋转位置编码。

    在 ``model_jit.Attention`` 中，它只作用于 RMSNorm 之后的 ``q`` 和 ``k``，不作用于 ``v``。JiT-B/16 中输入为 ``q/k:[B,12,N,64]``，
    其中``N=256``（16×16 image patch）或 ``N=288``（再加 32 个 context token），
    输出形状保持不变。二维 patch 网格的行、列位置被编码到每个 head 的 64 个
    feature 中，使 ``q·k`` 的相似度带有空间位置信息；context token 使用零旋转，因此不会被额外改变。
    """
    def __init__(
        self,
        dim,                    # 每个空间方向的基础频率维度；JiT-B: 32，行列拼接后旋转维度为 64。
        pt_seq_len=16,          # 预训练图像网格边长；JiT-B/16: 16（对应 16×16 个 patch）。
        ft_seq_len=None,        # 微调时的图像网格边长；None 表示沿用 pt_seq_len，即 16。
        custom_freqs = None,    # 用户自定义基础频率；非空时跳过 freqs_for 的默认频率构造。
        freqs_for = 'lang',     # 频率类型：lang 使用指数频率，pixel 使用线性频率，constant 使用常数频率。
        theta = 10000,          # lang 频率的基数；控制不同维度的位置变化速度。
        max_freq = 10,          # pixel 频率模式的最大频率参数；lang 模式下不使用。
        num_freqs = 1,          # constant 频率模式生成的频率数量；其他模式下不使用。
        num_cls_token = 0       # 前缀/context token 数；JiT-B 无 context 时为 0，含 context 时为 32。
    ):
        """预计算并注册 RoPE 的 cosine/sine 位置缓冲区。"""
        # 注册该位置编码模块。
        super().__init__()
        # 选择基础频率来源；custom_freqs 非空时直接使用外部频率。
        if custom_freqs:
            # 使用用户提供的频率；维度: [F] -> [F]。
            freqs = custom_freqs
        elif freqs_for == 'lang':
            # 生成语言式指数频率；JiT-B dim=32 时维度: [dim//2] -> [16]。freq_k=10000^(-k/10),k=0,1,...,15
            # 指数为 [0,1/16,...,15/16]，对应 freqs=10000^(-指数)。
            # 具体约为 [1,0.5623,0.3162,0.1778,0.1,0.0562,0.0316,0.0178,
            # 0.01,0.0056,0.0032,0.0018,0.001,0.00056,0.00032,0.00018]。
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            # 生成像素式线性频率；维度: [dim//2] -> [16]（dim=32）。
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            # 生成常数频率；维度: [num_freqs] -> [num_freqs]。
            freqs = torch.ones(num_freqs).float()
        else:
            # 拒绝未知频率类型；输出: 异常。
            raise ValueError(f'unknown modality {freqs_for}')

        # 未指定 fine-tune 序列长度时沿用预训练网格长度；标量: None -> 16。
        if ft_seq_len is None: ft_seq_len = pt_seq_len
        # 生成 fine-tune 网格的位置坐标，并缩放到预训练网格的坐标范围 [0, pt_seq_len)。
        # JiT-B 默认 ft_seq_len=pt_seq_len=16：arange(16)/16*16 -> [0,1,2,...,15]。
        # 若微调使用 ft_seq_len=32：arange(32)/32*16 -> [0,0.5,1,...,15.5]，用于分辨率变化时保持频率范围一致，并把位置采样点数从 16 个扩展为 32 个。
        # t: [ft_seq_len] -> [16]（默认）或 [32]（示例）。
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        # 位置坐标与基础频率外积；维度: [16] × [16] -> [16,16]。
        freqs = torch.einsum('..., f -> ... f', t, freqs)
        # 每个频率复制到二元旋转分量；维度: [16,16] -> [16,32]。
        freqs = repeat(freqs, '... n -> ... (n r)', r = 2)
        # 分别构造行方向和列方向频率，并拼接；维度: [16,1,32]、[1,16,32] -> [16,16,64]。
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim = -1)

        # 如果存在 context token，先把图像网格展平，再在前面添加不旋转的 context 位置。
        if num_cls_token > 0:
            # 展平图像网格；维度: [16,16,64] -> [256,64]。
            freqs_flat = freqs.view(-1, freqs.shape[-1])
            # 计算图像 token 的 cos/sin；维度: [256,64] -> [256,64]。
            cos_img = freqs_flat.cos()
            sin_img = freqs_flat.sin()

            # prepend in-context cls token
            # 读取图像 token 数和旋转维度；维度: [256,64] -> N_img=256, D=64。
            N_img, D = cos_img.shape
            # context token 的 cos 全为 1；维度: [num_cls_token,D] -> [32,64]。
            cos_pad = torch.ones(num_cls_token, D, dtype=cos_img.dtype, device=cos_img.device)
            # context token 的 sin 全为 0，表示不旋转；维度: [32,64]。
            sin_pad = torch.zeros(num_cls_token, D, dtype=sin_img.dtype, device=sin_img.device)

            # RoPE buffer 不在这里硬编码 .cuda()；由 Module.to(device) 随模型
            # 一起移动，这样 CPU smoke test 和多卡设备迁移都能正常工作。
            # 拼接 context 和 image 的 cos；维度: [32,64]+[256,64] -> [288,64]。
            self.register_buffer("freqs_cos", torch.cat([cos_pad, cos_img], dim=0))
            # 拼接 context 和 image 的 sin；维度: [32,64]+[256,64] -> [288,64]。
            self.register_buffer("freqs_sin", torch.cat([sin_pad, sin_img], dim=0))
        else:
            # 没有 context 时直接展平图像网格；维度: [16,16,64] -> [256,64]。
            self.register_buffer("freqs_cos", freqs.cos().view(-1, freqs.shape[-1]))
            # 保存对应的 sin 缓冲区；维度: [16,16,64] -> [256,64]。
            self.register_buffer("freqs_sin", freqs.sin().view(-1, freqs.shape[-1]))

    def forward(self, t):
        """对 q 或 k 应用二维 RoPE；输入/输出均为 ``[B,12,N,64]``。"""
        # 将 [N,64] 的 cos 广播到 batch/head；维度: [B,12,N,64]*[N,64] -> [B,12,N,64]。
        rotated_cos = t * self.freqs_cos
        # 旋转相邻 feature 对并乘以 sin；维度: [B,12,N,64]*[N,64] -> [B,12,N,64]。
        rotated_sin = rotate_half(t) * self.freqs_sin
        # 合并 cos/sin 两部分；维度: [B,12,N,64]+[B,12,N,64] -> [B,12,N,64]。
        return rotated_cos + rotated_sin


class RMSNorm(nn.Module):
    """对输入最后一维执行 Root Mean Square Normalization。

    计算公式为 ``y = x / sqrt(mean(x^2) + eps) * weight``，不减去均值。
    输入可以是 ``[B, ..., D]``；JiT-B attention 中为
    ``[B, num_heads, N, head_dim] -> [B, 12, 256/288, 64]``，
    ``mean(-1)`` 只归一化每个 head 的 64 个特征。Transformer token
    的 RMSNorm(768) 则处理 ``[B, N, 768]`` 的最后 768 个特征。
    """
    def __init__(self, hidden_size, eps=1e-6):
        """初始化可学习缩放权重和数值稳定项。"""
        # 注册 RMSNorm 模块及其参数。
        super().__init__()
        # 创建逐 feature 的可学习缩放权重；维度: [D] -> [64]（Q/K）或 [768]（Transformer token）。
        self.weight = nn.Parameter(torch.ones(hidden_size))
        # 保存防止除零的 epsilon；维度: 标量 -> 1e-6。
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        """沿最后一维归一化并恢复输入 dtype；形状保持 ``[B, ..., D]``。"""
        # 保存输入数据类型；dtype: bf16/fp16/fp32 -> 原 dtype。
        input_dtype = hidden_states.dtype
        # 用 float32 计算均方和归一化，提高半精度输入下的数值稳定性；维度保持 [B, ..., D]。
        hidden_states = hidden_states.to(torch.float32)
        # 只沿最后一维 D 计算均方值；[B,num_heads,N,D] -> [B,num_heads,N,1]。
        # JiT-B attention: [B,12,256/288,64] -> [B,12,256/288,1]。
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # 用 sqrt(mean(x^2)+eps) 的倒数缩放；维度: [B,num_heads,N,D] * [B,num_heads,N,1] -> [B,num_heads,N,D]。
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # 乘以可学习 weight 并恢复输入 dtype；[B,12,N,64] * [64] -> [B,12,N,64]，dtype -> input_dtype。
        return (self.weight * hidden_states).to(input_dtype)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
