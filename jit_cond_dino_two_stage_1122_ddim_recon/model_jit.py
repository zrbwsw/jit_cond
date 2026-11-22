# --------------------------------------------------------
# References:
# SiT: https://github.com/willisma/SiT
# Lightning-DiT: https://github.com/hustvl/LightningDiT
# --------------------------------------------------------
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from util.model_util import VisionRotaryEmbeddingFast, get_2d_sincos_pos_embed, RMSNorm


def modulate(x, shift, scale):
    """使用 AdaLN 公式对 token 特征进行条件调制。

    ``x`` 为 ``[B, N, D]``；``shift`` 和 ``scale`` 可以是全局条件
    ``[B, D]``，也可以是逐 patch 条件 ``[B, N, D]``。JiT-B/16 中
    ``D=768``，``N=256``，加入 32 个 in-context token 后 ``N=288``。
    返回值形状与 ``x`` 相同。
    """
    # 检查条件是否只有 batch 和 feature 两个维度；shift: [B, D] -> [B, 768]。
    if shift.ndim == 2:
        # 为全局 shift 增加 token 维；[B, 768] -> [B, 1, 768]。
        shift = shift.unsqueeze(1)
        # 为全局 scale 增加 token 维；[B, 768] -> [B, 1, 768]。
        scale = scale.unsqueeze(1)
    # 执行 x * (1 + scale) + shift；x/输出: [B, N, D] -> [B, 256/288, 768]。
    return x * (1 + scale) + shift


class BottleneckPatchEmbed(nn.Module):
    """把输入图像切成 patch，并映射到 Transformer hidden dimension。

    JiT-B/16 的输入 ``[B, 3, 256, 256]`` 经过 16×16 patch 切分后得到
    ``16×16=256`` 个 token，每个 token 的维度为 ``768``，输出为
    ``[B, 256, 768]``。
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, pca_dim=768, embed_dim=768, bias=True):
        """初始化两层卷积 patch embedding。"""
        # 注册该模块中的卷积层和参数。
        super().__init__()
        # 把标量图像尺寸扩展为 [H, W]；JiT-B/16: 256 -> [256, 256]。
        img_size = (img_size, img_size)
        # 把标量 patch 尺寸扩展为 [patch_h, patch_w]；16 -> [16, 16]。
        patch_size = (patch_size, patch_size)
        # 计算 patch 数；维度: 标量 -> (256/16)*(256/16)=256。
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        # 保存图像空间尺寸；维度: [H, W] -> [256, 256]。
        self.img_size = img_size
        # 保存 patch 空间尺寸；维度: [patch_h, patch_w] -> [16, 16]。
        self.patch_size = patch_size
        # 保存 patch 数量；维度: 标量 -> 256。
        self.num_patches = num_patches

        # 用 patch 大小卷积完成切 patch 和 bottleneck 投影；权重: [128, 3, 16, 16]。
        self.proj1 = nn.Conv2d(in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        # 用 1×1 卷积把 bottleneck channel 映射到 hidden size；权重: [768, 128, 1, 1]。
        self.proj2 = nn.Conv2d(pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        """把图像 ``[B, C, H, W]`` 转换为 token ``[B, N, D]``。"""
        # 读取输入形状；维度: [B, C, H, W] -> [B, 3, 256, 256]。
        B, C, H, W = x.shape
        # 确认输入空间尺寸和 patch embedding 配置一致；H/W: 256/256。
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # 依次执行两层卷积、展平空间维并转置；[B,3,256,256] -> [B,128,16,16] -> [B,768,16,16] -> [B,256,768]。
        x = self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)
        # 返回 patch token 序列；维度: [B, N, D] -> [B, 256, 768]。
        return x


class TimestepEmbedder(nn.Module):
    """把标量扩散时间步编码成 Transformer 条件向量。"""
    def __init__(self, hidden_size, frequency_embedding_size=256):
        """初始化正弦时间步编码和两层 MLP。"""
        # 注册 MLP 层和参数。
        super().__init__()
        # MLP 将频率特征从 256 维映射到 hidden size；[B, 256] -> [B, 768]。
        self.mlp = nn.Sequential(
            # 第一层线性投影；权重: [hidden_size, frequency_size] -> [768, 256]。
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            # SiLU 逐元素激活；维度: [B, 768] -> [B, 768]。
            nn.SiLU(),
            # 第二层保持 hidden size；权重: [768, 768]。
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        # 保存频率嵌入维度；维度: 标量 -> 256。
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """生成可处理连续时间步的正弦编码；``t: [B] -> embedding: [B, dim]``。"""
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        # 只构造一半频率，另一半由 cos/sin 拼接；维度: 标量 -> 128（dim=256）。
        half = dim // 2
        # 生成指数频率；维度: [half] -> [128]。
        freqs = torch.exp(
            # arange 生成 0 到 half-1；维度: [128] -> [128]。
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        # 将频率移动到时间步所在设备；维度: [128] -> [128]。
        ).to(device=t.device)
        # 时间步与每个频率相乘；维度: [B, 1] * [1, 128] -> [B, 128]。
        args = t[:, None].float() * freqs[None]
        # 拼接 cos 和 sin；维度: [B, 128] + [B, 128] -> [B, 256]。
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        # 只有 dim 为奇数时补零；JiT-B 的 dim=256，不执行此分支。
        if dim % 2:
            # 补一个维度；维度: [B, dim] -> [B, dim+1]。
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        # 返回时间步频率编码；维度: [B, 256] -> [B, 256]。
        return embedding

    def forward(self, t):
        """把时间步 ``[B]`` 转换为 hidden 条件 ``[B, 768]``。"""
        # 生成正弦频率编码；维度: [B] -> [B, 256]。
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        # 通过 MLP 映射到 hidden size；维度: [B, 256] -> [B, 768]。
        t_emb = self.mlp(t_freq)
        # 返回时间条件向量；维度: [B, 768] -> [B, 768]。
        return t_emb


class LabelEmbedder(nn.Module):
    """把类别索引映射为 class-condition 向量。"""
    def __init__(self, num_classes, hidden_size):
        """初始化类别 embedding 表；额外一行用于 classifier-free guidance。"""
        # 注册 embedding 参数。
        super().__init__()
        # 建立类别查找表；权重: [num_classes+1, hidden_size] -> [1001, 768]。
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        # 保存类别数量；维度: 标量 -> 1000。
        self.num_classes = num_classes

    def forward(self, labels):
        """查表得到类别条件；输入 ``[B]``，输出 ``[B, 768]``。"""
        # 根据每个样本的类别索引查表；维度: [B] -> [B, 768]。
        embeddings = self.embedding_table(labels)
        # 返回类别 embedding；维度: [B, 768] -> [B, 768]。
        return embeddings


def scaled_dot_product_attention(query, key, value, dropout_p=0.0) -> torch.Tensor:
    """计算缩放点积注意力（Scaled Dot-Product Attention）。

    对每个 query token，根据它与所有 key token 的相似度计算权重，
    再对 value token 做加权求和：

        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    ``query: [B, num_heads, L, head_dim]``
    ``key: [B, num_heads, S, head_dim]``
    ``value: [B, num_heads, S, value_dim]``
    其中 L 是 query token 数，S 是 key/value token 数。返回张量形状为``[B, num_heads, L, value_dim]``。
    """
    # JiT-B/16: 256x256 图像切成 16x16 个 patch；加入 32 个 context token 后，序列长度为 288。
    # L: query token 数 -> 256（无 context）/288（含 32 context）；S: key/value token 数 -> 256/288。
    # 读取 query 和 key 的 token 维度；L/S: 标量 -> 256（无 context）或 288（含 context）。
    L, S = query.size(-2), key.size(-2)

    # 除以 sqrt(head_dim) 可以抑制点积数值随维度变大，避免 softmax 过早饱和。
    # scale_factor: 标量 -> 1/sqrt(64)=0.125（JiT-B: 768/12=64）
    scale_factor = 1 / math.sqrt(query.size(-1))

    # 当前没有额外的 attention mask，因此使用全零 bias；第二维为 1，会广播到 12 个 attention heads。
    # attn_bias: [B, 1, L, S] -> [B, 1, 256, 256]（无 context）/ [B, 1, 288, 288]（含 context）
    attn_bias = torch.zeros(query.size(0), 1, L, S, dtype=query.dtype, device=query.device)

    # QK^T 先转为 float32 计算，降低半精度下矩阵乘法和 softmax 的数值误差。
    # 禁用当前作用域的 CUDA 自动混合精度，确保 QK^T 在显式 float32 中计算。
    with torch.amp.autocast("cuda", enabled=False):
        # query: [B, num_heads, L, head_dim] -> [B, 12, 256, 64] / [B, 12, 288, 64]；
        # key^T: [B, 12, 64, 256] / [B, 12, 64, 288]；
        # attn_weight: [B, 12, 256, 256] / [B, 12, 288, 288]
        # 计算缩放点积；维度: [B,12,L,64] @ [B,12,64,S] -> [B,12,L,S]。
        attn_weight = query.float() @ key.float().transpose(-2, -1) * scale_factor

    # 对每个 query，在 S 个 key 上归一化得到注意力权重。
    # 加入 attention bias；维度: [B, num_heads, L, S] + [B, 1, L, S] -> [B, 12, 256/288, 256/288]。
    attn_weight += attn_bias
    # attn_weight: [B, num_heads, L, S] -> [B, 12, 256, 256]（无 context）/ [B, 12, 288, 288]（含 context）
    # 沿最后的 key 维 S 做 softmax；维度: [B, 12, L, S] -> [B, 12, L, S]。
    attn_weight = torch.softmax(attn_weight, dim=-1)

    # 按 dropout_p 随机丢弃部分注意力权重；推理时调用方传入 0，因此不会丢弃。
    # attn_weight: [B, num_heads, L, S] -> [B, 12, 256, 256]（无 context）/ [B, 12, 288, 288]（含 context）
    # 对注意力权重做 dropout；维度保持 [B, 12, L, S] -> [B, 12, L, S]。
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)

    # 用注意力权重对 value 做加权求和，得到每个 query token 的输出表示。
    # output: [B, num_heads, L, value_dim] -> [B, 12, 256, 64]（无 context）/ [B, 12, 288, 64]（含 context）
    # 加权汇总 value；维度: [B, 12, L, S] @ [B, 12, S, value_dim] -> [B, 12, L, value_dim]。
    return attn_weight @ value


class Attention(nn.Module):
    """执行多头自注意力，并把多头结果投影回 hidden dimension。"""
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0., proj_drop=0.):
        """初始化 QKV 投影、Q/K 归一化、attention dropout 和输出投影。"""
        # 注册该 attention 模块。
        super().__init__()
        # 保存 head 数；JiT-B: 标量 -> 12。
        self.num_heads = num_heads
        # 计算每个 head 的维度；JiT-B: 768/12 -> 64。
        head_dim = dim // num_heads

        # 对 query 做 RMSNorm；维度: [B, num_heads, N, head_dim] -> [B, 12, N, 64]。
        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        # 对 key 做 RMSNorm；维度: [B, num_heads, N, head_dim] -> [B, 12, N, 64]。
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        # 一次线性层同时生成 Q/K/V；输入/输出: [B, N, 768] -> [B, N, 2304]。
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # 保存 attention 权重 dropout；输入/输出维度保持 [B, 12, N, N]。
        self.attn_drop = nn.Dropout(attn_drop)
        # 多头拼接后的输出投影；输入/输出: [B, N, 768] -> [B, N, 768]。
        self.proj = nn.Linear(dim, dim)
        # 保存输出 dropout；维度保持 [B, N, 768]。
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope):
        """对 token 序列 ``x`` 执行多头自注意力。

        ``x`` 为 ``[B, N, D]``，JiT-B/16 中为 ``[B, 256/288, 768]``；
        返回形状不变。
        """
        # 读取输入形状；维度: [B, N, C] -> [B, 256/288, 768]。
        B, N, C = x.shape
        # 生成并拆分 QKV；维度: [B,N,768] -> [B,N,2304] -> [B,N,3,12,64] -> [3,B,12,N,64]。
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # 取出 query/key/value；维度: [3,B,12,N,64] -> 每个 [B,12,N,64]。
        # 分别取出 Q/K/V；这样写可兼容 TorchScript，维度: [3,B,12,N,64] -> 3 个 [B,12,N,64]。
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 对 query 做归一化；维度: [B,12,N,64] -> [B,12,N,64]。
        q = self.q_norm(q)
        # 对 key 做归一化；维度: [B,12,N,64] -> [B,12,N,64]。
        k = self.k_norm(k)

        # 对 query 应用二维旋转位置编码；维度: [B,12,N,64] -> [B,12,N,64]。
        q = rope(q)
        # 对 key 应用二维旋转位置编码；维度: [B,12,N,64] -> [B,12,N,64]。
        k = rope(k)

        # 计算缩放点积注意力；维度: [B,12,N,64] -> [B,12,N,64]。
        x = scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)

        # 把 head 维移回 token 后面并拼接 heads；维度: [B,12,N,64] -> [B,N,768]。
        x = x.transpose(1, 2).reshape(B, N, C)

        # 对拼接后的多头表示做输出投影；维度: [B,N,768] -> [B,N,768]。
        x = self.proj(x)
        # 应用输出 dropout；维度: [B,N,768] -> [B,N,768]。
        x = self.proj_drop(x)
        # 返回 attention 输出；维度: [B,N,768] -> [B,256/288,768]。
        return x


class SwiGLUFFN(nn.Module):
    """使用 SwiGLU 激活的前馈网络，把每个 token 独立映射回 hidden size。"""
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        drop=0.0,
        bias=True
    ) -> None:
        """初始化 SwiGLU 的扩展投影、门控投影和 dropout。"""
        # 注册该前馈模块。
        super().__init__()
        # SwiGLU 使用 2/3 缩放隐藏层；JiT-B: 768*4*2/3=2048。
        hidden_dim = int(hidden_dim * 2 / 3)
        # 一次投影生成两路门控输入；输入/输出: [B,N,768] -> [B,N,4096]。
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        # 把门控后的 hidden 表示映射回 model dimension；[B,N,2048] -> [B,N,768]。
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        # 保存 FFN dropout；维度保持 [B,N,2048]。
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        """执行 SwiGLU 前馈变换；输入/输出均为 ``[B, N, 768]``。"""
        # 生成两路门控特征；维度: [B,N,768] -> [B,N,4096]。
        x12 = self.w12(x)
        # 沿最后一维拆成两路；维度: [B,N,4096] -> 两个 [B,N,2048]。
        x1, x2 = x12.chunk(2, dim=-1)
        # 计算 SiLU(x1) * x2；维度: [B,N,2048] -> [B,N,2048]。
        hidden = F.silu(x1) * x2
        # dropout 后投影回 hidden size；维度: [B,N,2048] -> [B,N,768]。
        return self.w3(self.ffn_dropout(hidden))


class FinalLayer(nn.Module):
    """把 Transformer token 还原为每个图像 patch 的像素预测。"""
    def __init__(self, hidden_size, patch_size, out_channels):
        """初始化最终 RMSNorm、AdaLN 调制和 patch 像素投影。"""
        # 注册该输出层。
        super().__init__()
        # 对 hidden token 做 RMSNorm；维度保持 [B,N,768]。
        self.norm_final = RMSNorm(hidden_size)
        # 每个 token 预测一个 patch 的所有像素；JiT-B/16: 768 -> 16*16*3=768。
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        # 由条件向量生成 shift/scale；全局/patch 条件分别为 [B,768]/[B,N,768] -> [B,1536]/[B,N,1536]。
        self.adaLN_modulation = nn.Sequential(
            # SiLU 逐元素激活；维度: [B,768] 或 [B,N,768] -> 不变。
            nn.SiLU(),
            # 生成两组调制参数；维度: [B,768] 或 [B,N,768] -> [B,1536] 或 [B,N,1536]。
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    @torch.compile
    def forward(self, x, c, patch_condition=None):
        """使用条件调制后预测 patch 像素。

        ``x`` 为 ``[B,N,768]``，``c`` 为 ``[B,768]``；启用 grounding 时
        ``patch_condition`` 为 ``[B,N,768]``。输出为 ``[B,N,768]``，其中
        每个 768 维向量对应一个 16×16×3 patch。
        """
        # 有 patch 条件时逐 patch 调制，否则使用全局条件；condition: [B,768] 或 [B,N,768]。
        condition = c if patch_condition is None else patch_condition
        # 生成 shift 和 scale；维度: [B,1536]/[B,N,1536] -> 两个 [B,768]/[B,N,768]。
        shift, scale = self.adaLN_modulation(condition).chunk(2, dim=-1)
        # 先归一化再做 AdaLN 调制；维度: [B,N,768] -> [B,N,768]。
        x = modulate(self.norm_final(x), shift, scale)
        # 每个 token 预测一个 patch 的像素和通道；维度: [B,N,768] -> [B,N,768]。
        x = self.linear(x)
        # 返回 patch 预测；维度: [B,N,768] -> [B,256/288,768]。
        return x


class JiTBlock(nn.Module):
    """一个带 AdaLN 条件调制的 Transformer block。"""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        """初始化 RMSNorm、attention、SwiGLU FFN 和六路 AdaLN 参数。"""
        # 注册该 Transformer block。
        super().__init__()
        # attention 分支前的 RMSNorm；维度保持 [B,N,768]。
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        # 多头自注意力；JiT-B: 12 heads、每头 64 维。
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop)
        # FFN 分支前的 RMSNorm；维度保持 [B,N,768]。
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        # 计算 FFN 的初始隐藏维度；JiT-B: 768*4=3072，SwiGLU 内部缩放为 2048。
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # 初始化 SwiGLU FFN；输入/输出: [B,N,768] -> [B,N,768]。
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        # 生成 attention/FFN 各自的 shift、scale、gate 共 6 组参数；输出: [B,D] -> [B,4608]。
        self.adaLN_modulation = nn.Sequential(
            # SiLU 逐元素激活；维度: [B,D] 或 [B,N,D] -> 不变。
            nn.SiLU(),
            # 六路调制参数；维度: [B,768] 或 [B,N,768] -> [B,4608] 或 [B,N,4608]。
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    @torch.compile
    def forward(self, x, c, feat_rope=None, patch_condition=None, num_prefix_tokens=0):
        """执行一次条件 Transformer 更新并返回同形状 token。

        ``x`` 为 ``[B,N,768]``；``c`` 为全局条件 ``[B,768]``；grounding
        条件为 ``[B,256,768]``，加入 32 个 prefix 后可扩展为 ``[B,288,768]``。
        """
        # image patch 使用 [B,N,D] 的 patch condition；JiT 的 in-context
        # class token 没有对应的 DINO patch，因此继续使用全局 t+y。
        # 没有 patch 条件时，所有 token 共用全局条件；condition: [B,768]。
        if patch_condition is None:
            condition = c
        # 有 prefix 且存在 patch 条件时，为 prefix 复制全局条件并拼接 patch 条件。
        elif num_prefix_tokens:
            # 增加 token 维并复制到 prefix 数量；维度: [B,768] -> [B,1,768] -> [B,32,768]。
            prefix = c.unsqueeze(1).expand(-1, num_prefix_tokens, -1)
            # 拼接 prefix 和 image patch 条件；维度: [B,32,768]+[B,256,768] -> [B,288,768]。
            condition = torch.cat((prefix, patch_condition), dim=1)
        # 没有 prefix 时直接使用逐 patch 条件；维度: [B,256,768]。
        else:
            condition = patch_condition
        # 生成六路 AdaLN 参数；维度: [B,4608]/[B,N,4608] -> 6 个 [B,768]/[B,N,768]。
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(condition).chunk(6, dim=-1)
        # global c 产生 [B,D] gate，需要扩展到 token 维；patch condition
        # 已经是 [B,N,D]，不能再次 unsqueeze，否则会多出一维。
        # 判断 gate 是否只有 batch 和 feature 维；全局 gate: [B,768]。
        if gate_msa.ndim == 2:
            # 扩展 attention gate 的 token 维；[B,768] -> [B,1,768]。
            gate_msa = gate_msa.unsqueeze(1)
            # 扩展 FFN gate 的 token 维；[B,768] -> [B,1,768]。
            gate_mlp = gate_mlp.unsqueeze(1)
        # attention 残差：调制/归一化后 attention 输出仍为 [B,N,768]。
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        # FFN 残差：调制/归一化后 FFN 输出仍为 [B,N,768]。
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        # 返回 block 输出；维度: [B,N,768] -> [B,256/288,768]。
        return x


class JiT(nn.Module):
    """JiT 图像 Transformer，负责条件编码、token 处理和 patch 重建。"""
    def __init__(
        self,
        input_size=256,
        patch_size=16,
        in_channels=3,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        num_classes=1000,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=8,
        grounding_dim=None,
        use_t_dino=False,
    ):
        """初始化 JiT 主干。

        JiT-B/16 的典型配置为 ``input_size=256``、``patch_size=16``、
        ``hidden_size=768``、``depth=12``、``num_heads=12``；输入为
        ``[B,3,256,256]``，最终输出同样为 ``[B,3,256,256]``。
        """
        # 注册 JiT 主模块。
        super().__init__()
        # 保存输入通道数；维度: 标量 -> 3。
        self.in_channels = in_channels
        # 输出通道与输入通道一致；维度: 标量 -> 3。
        self.out_channels = in_channels
        # 保存 patch 边长；维度: 标量 -> 16。
        self.patch_size = patch_size
        # 保存 attention head 数；维度: 标量 -> 12。
        self.num_heads = num_heads
        # 保存 Transformer hidden size；维度: 标量 -> 768。
        self.hidden_size = hidden_size
        # 保存输入图像边长；维度: 标量 -> 256。
        self.input_size = input_size
        # 保存 in-context token 数；维度: 标量 -> 32。
        self.in_context_len = in_context_len
        # 保存插入 in-context token 的 block 下标；JiT-B: 标量 -> 4。
        self.in_context_start = in_context_start
        # 保存类别数；维度: 标量 -> 1000。
        self.num_classes = num_classes
        # grounding_dim=None 保留原始 JiT API；设置为 768 后启用 DINOv3
        # patch token 投影。投影层是唯一新增的 JiT 可训练参数模块。
        # 保存 grounding 输入维度；维度: None 或标量 -> None/768。
        self.grounding_dim = grounding_dim
        # 保留 join-dino 的参数结构以兼容旧权重；本版本预测 clean token，
        # 训练与采样均令 Pixel 的 t_dino=1，实际加噪时间只送给 DinoDiT。
        self.use_t_dino = bool(use_t_dino)

        # time and class embed
        # 初始化时间步条件编码器；输入/输出: [B] -> [B,768]。
        self.t_embedder = TimestepEmbedder(hidden_size)
        if self.use_t_dino:
            # grounding 的噪声时间 t_dino -> [B,768]，与 t_embedder 不共享权重。
            self.t_dino_embedder = TimestepEmbedder(hidden_size)
        # 初始化类别条件编码器；输入/输出: [B] -> [B,768]。
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)

        # 只有启用 grounding 时才创建 DINO token 投影层。
        if self.grounding_dim is not None:
            # DINOv3-B/16 token 维度为 768，映射到 JiT hidden_size=768。
            # 该层对每个 patch 独立作用，不随 patch 数量增加参数。
            # grounding token 投影；输入/输出: [B,N_g,768] -> [B,N_g,768]。
            self.grounding_proj = nn.Linear(self.grounding_dim, hidden_size, bias=True)

        # linear embed
        # 初始化图像 patch embedding；[B,3,256,256] -> [B,256,768]。
        self.x_embedder = BottleneckPatchEmbed(input_size, patch_size, in_channels, bottleneck_dim, hidden_size, bias=True)

        # use fixed sin-cos embedding
        # 读取 patch 数；维度: 标量 -> 256。
        num_patches = self.x_embedder.num_patches
        # 创建冻结的二维正弦位置编码参数；维度: [1,N,D] -> [1,256,768]。
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # in-context cls token
        # 只有配置了 context token 时才创建可学习位置编码。
        if self.in_context_len > 0:
            # context 位置编码；维度: [1,32,768]。
            self.in_context_posemb = nn.Parameter(torch.zeros(1, self.in_context_len, hidden_size), requires_grad=True)
            # 初始化 context 位置编码；维度保持 [1,32,768]。
            torch.nn.init.normal_(self.in_context_posemb, std=.02)

        # rope
        # 计算每个 head 的旋转位置编码维度；JiT-B: 768/12/2=32。
        half_head_dim = hidden_size // num_heads // 2
        # 计算 patch 网格边长；JiT-B/16: 256/16=16。
        hw_seq_len = input_size // patch_size
        # 为纯 image token 创建 RoPE；token 数: 256，class token 数: 0。
        self.feat_rope = VisionRotaryEmbeddingFast(
            # RoPE 维度: 标量 -> 32。
            dim=half_head_dim,
            # 图像网格边长: 标量 -> 16。
            pt_seq_len=hw_seq_len,
            # 不包含 context token；维度: 标量 -> 0。
            num_cls_token=0
        )
        # 为包含 in-context token 的序列创建 RoPE；token 数: 256+32=288。
        self.feat_rope_incontext = VisionRotaryEmbeddingFast(
            # RoPE 维度: 标量 -> 32。
            dim=half_head_dim,
            # 图像网格边长: 标量 -> 16。
            pt_seq_len=hw_seq_len,
            # 预留 context token 数；维度: 标量 -> 32。
            num_cls_token=self.in_context_len
        )

        # transformer
        # 创建 depth 个 Transformer block；JiT-B: depth=12，每个 block 处理 [B,256/288,768]。
        self.blocks = nn.ModuleList([
            # 初始化第 i 个 block；配置输出维度保持 [B,N,768]。
            JiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                     # 仅在指定 block 区间启用 attention dropout；标量概率 -> attn_drop 或 0。
                     attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                     # 仅在指定 block 区间启用 FFN/projection dropout；标量概率 -> proj_drop 或 0。
                     proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0)
            for i in range(depth)
        ])

        # linear predict
        # 初始化最终 patch 预测层；[B,N,768] -> [B,N,16*16*3=768]。
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        # 初始化所有线性层、位置编码和零初始化调制层。
        self.initialize_weights()

    def initialize_weights(self):
        """初始化线性层、位置编码、embedding 和零初始条件调制层。"""
        # 定义递归应用到所有子模块的基础线性层初始化函数。
        def _basic_init(module):
            """用 Xavier 初始化线性层权重，并将 bias 初始化为零。"""
            # 只处理 nn.Linear，卷积和其他模块使用后面的专门初始化逻辑。
            if isinstance(module, nn.Linear):
                # Xavier 初始化线性权重；维度: [out_features, in_features] -> 不变。
                torch.nn.init.xavier_uniform_(module.weight)
                # 线性层存在 bias 时清零；维度: [out_features] -> [out_features]。
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        # 将基础初始化递归应用到当前模型所有子模块。
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        # 生成二维正弦位置编码；维度: [N, D] -> [256, 768]。
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        # 转成 tensor、增加 batch 维并复制到冻结参数；维度: [256,768] -> [1,256,768]。
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        # 读取第一层 patch 卷积权重；维度: [128,3,16,16] -> [128,3,16,16]。
        w1 = self.x_embedder.proj1.weight.data
        # 把卷积核展平后做 Xavier 初始化；维度: [128,3,16,16] -> [128,768]。
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        # 读取第二层 1×1 卷积权重；维度: [768,128,1,1] -> [768,128,1,1]。
        w2 = self.x_embedder.proj2.weight.data
        # 把 1×1 卷积权重展平后初始化；维度: [768,128,1,1] -> [768,128]。
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        # 清零第二层卷积 bias；维度: [768] -> [768]。
        nn.init.constant_(self.x_embedder.proj2.bias, 0)

        # Initialize label embedding table:
        # 初始化类别 embedding；维度: [1001,768] -> [1001,768]。
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # 初始化时间步 MLP 第一层权重；维度: [768,256] -> [768,256]。
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        # 初始化时间步 MLP 第二层权重；维度: [768,768] -> [768,768]。
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        if self.use_t_dino:
            nn.init.normal_(self.t_dino_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_dino_embedder.mlp[2].weight, std=0.02)

        # Zero-out the grounding projection so the grounded model starts at
        # exactly the baseline function: W_g @ dino == 0 makes
        # patch_condition == c broadcast to every patch.  Xavier init would
        # instead give the semantic term ~98% of the condition variance and
        # bury the t+y signal from step 0.
        # 只有 grounding 模型才把投影层权重和 bias 清零；权重: [768,768] -> [768,768]。
        if self.grounding_dim is not None:
            # 让初始 grounding 分支输出全零，保持与 baseline 相同；维度: [768,768] -> [768,768]。
            nn.init.constant_(self.grounding_proj.weight, 0)
            # grounding bias 清零；维度: [768] -> [768]。
            nn.init.constant_(self.grounding_proj.bias, 0)

        # Zero-out adaLN modulation layers:
        # 遍历所有 Transformer block；循环次数: depth -> 12。
        for block in self.blocks:
            # 清零 block 的六路调制权重；维度: [6*768,768] -> [4608,768]。
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            # 清零 block 的六路调制 bias；维度: [4608] -> [4608]。
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        # 清零最终层的 shift/scale 调制权重；维度: [2*768,768] -> [1536,768]。
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        # 清零最终层的 shift/scale 调制 bias；维度: [1536] -> [1536]。
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

        # 清零最终 patch 预测权重；JiT-B/16: [768,768] -> [768,768]。
        nn.init.constant_(self.final_layer.linear.weight, 0)
        # 清零最终 patch 预测 bias；维度: [768] -> [768]。
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, p):
        """把每个 patch 的像素预测重新排列为图像。

        输入 ``x`` 为 ``[B, N, p*p*C]``；JiT-B/16 中为 ``[B,256,768]``，
        输出图像为 ``[B,3,256,256]``。
        """
        # 保存输出通道数；维度: 标量 -> 3。
        c = self.out_channels
        # 根据 token 数恢复正方形 patch 网格；维度: 标量 -> h=w=16。
        h = w = int(x.shape[1] ** 0.5)
        # 确认 token 数能组成正方形；h*w: 标量 -> 256。
        assert h * w == x.shape[1]

        # 将每个 token 拆成 patch 网格和 patch 内部坐标；[B,256,768] -> [B,16,16,16,16,3]。
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        # 调整维度顺序，把 channel 放到第二维；[B,16,16,16,16,3] -> [B,3,16,16,16,16]。
        x = torch.einsum('nhwpqc->nchpwq', x)
        # 合并网格坐标和 patch 坐标恢复图像；[B,3,16,16,16,16] -> [B,3,256,256]。
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        # 返回重建图像；维度: [B,3,256,256] -> [B,3,256,256]。
        return imgs

    def _prepare_grounding(self, grounding_tokens, num_patches, grounding_mask=None):
        """投影 DINO token，并在需要时调整到 JiT 的 patch 网格。

        ``grounding_mask`` 是 [B] 的 bool；为 False 的样本语义项被清零，
        于是它的 patch_condition 退化为广播后的全局 ``c``。这与
        ``grounding_tokens=None`` 在数值上完全等价，但保持了 [B,N,D] 的
        形状，编译后不会因为 batch 内混合两种条件而产生两条图。

        输入 grounding token 为 ``[B, N_g, 768]``，输出对齐到 JiT patch 数
        ``[B, N, 768]``；JiT-B/16 中 ``N=256``。
        """
        # 没有提供 grounding token 时不创建额外条件；返回: None。
        if grounding_tokens is None:
            return None
        # 整批关闭语义条件（CFG 的无条件分支）：直接返回 None，省掉一次
        # [B,N,768]->[B,N,hidden] 的投影与插值；结果与置零完全等价。
        # 整批关闭 grounding 时直接返回 None；输入 mask: bool -> 返回 None。
        if isinstance(grounding_mask, bool) and not grounding_mask:
            return None
        # 检查模型是否配置了 grounding 投影维度；grounding_dim: None 或标量。
        if self.grounding_dim is None:
            raise ValueError("grounding tokens were supplied but grounding_dim is None")
        # 检查 grounding token 是否为三维且最后一维等于 configured grounding_dim；输入: [B,N_g,D_g]。
        if grounding_tokens.ndim != 3 or grounding_tokens.shape[-1] != self.grounding_dim:
            raise ValueError(
                f"grounding_tokens must have shape [B,N,{self.grounding_dim}], "
                f"got {tuple(grounding_tokens.shape)}"
            )
        # 先把 DINO 语义 token 投影到 JiT hidden_size，再与 t+y 相加。
        # grounding 投影；维度: [B,N_g,768] -> [B,N_g,768]。
        tokens = self.grounding_proj(grounding_tokens)
        # 当输入 token 数与 JiT patch 数不同时，需要把 token 网格插值到目标网格。
        if tokens.shape[1] != num_patches:
            # 推断源网格边长；维度: 标量 N_g -> sqrt(N_g)，如 256 -> 16。
            src_hw = int(tokens.shape[1] ** 0.5)
            # 推断目标网格边长；维度: 标量 N -> sqrt(N)，JiT-B/16: 256 -> 16。
            dst_hw = int(num_patches ** 0.5)
            # 检查源/目标 token 数都能组成正方形网格；维度: 标量判断。
            if src_hw * src_hw != tokens.shape[1] or dst_hw * dst_hw != num_patches:
                raise ValueError("JiT and grounding token counts must form square patch grids")
            # 把 token 序列转为 feature map；维度: [B,N_g,768] -> [B,768,src_hw,src_hw]。
            tokens = tokens.transpose(1, 2).reshape(tokens.shape[0], self.hidden_size, src_hw, src_hw)
            # 双线性插值到 JiT patch 网格；维度: [B,768,src_hw,src_hw] -> [B,768,16,16]。
            tokens = F.interpolate(tokens, size=(dst_hw, dst_hw), mode="bilinear", align_corners=False)
            # 再展平并转回 token 排列；维度: [B,768,16,16] -> [B,256,768]。
            tokens = tokens.flatten(2).transpose(1, 2)
        # 只有提供 mask 时才按样本控制 grounding 条件。
        if grounding_mask is not None:
            # 处理整个 batch 共用的 bool mask。
            if isinstance(grounding_mask, bool):
                # 整个批次都关掉语义条件（CFG 无条件分支）。
                # bool=False 时清零所有 grounding token；维度: [B,256,768] -> [B,256,768]。
                if not grounding_mask:
                    tokens = torch.zeros_like(tokens)
            # 处理每个样本独立的 [B] mask。
            else:
                # 扩展 mask 以便广播到 token 和 feature；维度: [B] -> [B,1,1]。
                keep = grounding_mask.to(device=tokens.device, dtype=tokens.dtype).view(-1, 1, 1)
                # 按样本保留或清零 grounding；维度: [B,1,1]*[B,256,768] -> [B,256,768]。
                tokens = tokens * keep
        # 返回对齐后的 grounding token；维度: [B,N,768] -> [B,256,768]。
        return tokens

    def forward(self, x, t, y, grounding_tokens=None, grounding_mask=None, t_dino=None):
        """执行 JiT 完整前向过程并输出重建图像。

        输入：图像 ``x:[B,3,256,256]``、时间步 ``t:[B]``、类别 ``y:[B]``，
        可选 grounding token ``[B,N_g,768]`` 和 mask ``[B]``。输出为
        ``[B,3,256,256]``；中间 image token 为 ``[B,256,768]``，插入
        32 个 in-context token 后为 ``[B,288,768]``。
        """
        # t+y 仍然是每个样本的全局条件；grounding 只给 image patch 提供
        # 空间变化的附加条件，最终得到 c_patch[b,n]=t+y+g[b,n]。
        # 编码扩散时间步；维度: [B] -> [B,768]。
        t_emb = self.t_embedder(t)
        # 编码类别标签；维度: [B] -> [B,768]。
        y_emb = self.y_embedder(y)
        # 合并时间和类别条件；维度: [B,768]+[B,768] -> [B,768]。
        c = t_emb + y_emb
        if self.use_t_dino and grounding_tokens is not None and t_dino is not None \
                and not (isinstance(grounding_mask, bool) and not grounding_mask):
            # t_dino 只和 grounding 一起出现：被 mask 的样本（CFG 无条件分支）
            # 退回纯 t+y，与 semantic 项的屏蔽规则完全一致。
            t_dino_emb = self.t_dino_embedder(t_dino)
            if grounding_mask is not None and not isinstance(grounding_mask, bool):
                t_dino_emb = t_dino_emb * grounding_mask.to(dtype=t_dino_emb.dtype).view(-1, 1)
            c = c + t_dino_emb

        # forward JiT
        # 把图像切成 patch token；维度: [B,3,256,256] -> [B,256,768]。
        x = self.x_embedder(x)
        # 加入冻结的二维位置编码；维度: [B,256,768]+[1,256,768] -> [B,256,768]。
        x += self.pos_embed
        # 默认不使用逐 patch 条件；变量状态: None。
        patch_condition = None
        # 只有传入 grounding token 时才准备语义条件。
        if grounding_tokens is not None:
            # 对齐 grounding token 的 patch 数和 hidden size；维度: [B,N_g,768] -> [B,256,768] 或 None。
            semantic = self._prepare_grounding(grounding_tokens, x.shape[1], grounding_mask)
            # semantic 为 None 表示整批关闭语义条件（CFG 无条件分支），
            # 此时退回全局 c，与训练中被屏蔽样本的路径完全一致。
            # grounding 有效时，把全局条件与逐 patch 语义条件相加；维度: [B,1,768]+[B,256,768] -> [B,256,768]。
            if semantic is not None:
                patch_condition = c.unsqueeze(1) + semantic
                # DINO token 通常为 fp32，而训练时 JiT 可能在 bf16 autocast 中；
                # 这里对齐到 image token dtype，避免 AdaLN 条件触发额外类型转换。
                # 对齐条件和 image token 的数据类型；维度保持 [B,256,768]。
                patch_condition = patch_condition.to(dtype=x.dtype)

        # 依次执行 depth 个 Transformer block；JiT-B: 循环 12 次。
        for i, block in enumerate(self.blocks):
            # in-context
            # 在指定 block 前插入 context token；JiT-B: i=4 时插入 32 个。
            if self.in_context_len > 0 and i == self.in_context_start:
                # 复制类别条件作为 context token；维度: [B,768] -> [B,32,768]。
                in_context_tokens = y_emb.unsqueeze(1).repeat(1, self.in_context_len, 1)
                # 加入可学习 context 位置编码；维度: [B,32,768]+[1,32,768] -> [B,32,768]。
                in_context_tokens += self.in_context_posemb
                # 把 context 和 image token 拼接；维度: [B,32,768]+[B,256,768] -> [B,288,768]。
                x = torch.cat([in_context_tokens, x], dim=1)
            # 运行当前 Transformer block；维度: [B,256/288,768] -> [B,256/288,768]。
            x = block(
                # 传入当前 token 序列；维度: [B,256/288,768]。
                x,
                # 传入全局时间/类别条件；维度: [B,768]。
                c,
                # 根据是否加入 context 选择对应 RoPE；不改变 token 主形状。
                self.feat_rope if i < self.in_context_start else self.feat_rope_incontext,
                # 传入可选逐 patch 条件；维度: None 或 [B,256,768]。
                patch_condition=patch_condition,
                # context block 使用 32 个 prefix，其他 block 使用 0 个。
                num_prefix_tokens=self.in_context_len if i >= self.in_context_start else 0,
            )

        # 去除末尾的 context token，只保留 image patch token；维度: [B,288,768] -> [B,256,768]。
        x = x[:, self.in_context_len:]

        # 将 token 通过最终 AdaLN/Linear 层映射为 patch 像素；维度: [B,256,768] -> [B,256,768]。
        x = self.final_layer(x, c, patch_condition=patch_condition)
        # 将 patch 像素重新排列为图像；维度: [B,256,768] -> [B,3,256,256]。
        output = self.unpatchify(x, self.patch_size)

        # 返回模型输出图像；维度: [B,3,256,256] -> [B,3,256,256]。
        return output


def JiT_B_16(**kwargs):
    """构造 JiT-B/16：12 blocks、768 hidden、12 heads、16×16 patch。"""
    # 返回 JiT-B/16；256×256 输入时 token 序列为 [B,256,768]。
    return JiT(depth=12, hidden_size=768, num_heads=12,
               bottleneck_dim=128, in_context_len=32, in_context_start=4, patch_size=16, **kwargs)

def JiT_B_32(**kwargs):
    """构造 JiT-B/32：12 blocks、768 hidden、12 heads、32×32 patch。"""
    # 返回 JiT-B/32；256×256 输入时 token 数为 8×8=64。
    return JiT(depth=12, hidden_size=768, num_heads=12,
               bottleneck_dim=128, in_context_len=32, in_context_start=4, patch_size=32, **kwargs)

def JiT_L_16(**kwargs):
    """构造 JiT-L/16：24 blocks、1024 hidden、16 heads、16×16 patch。"""
    # 返回 JiT-L/16；256×256 输入时 token 数为 16×16=256。
    return JiT(depth=24, hidden_size=1024, num_heads=16,
               bottleneck_dim=128, in_context_len=32, in_context_start=8, patch_size=16, **kwargs)

def JiT_L_32(**kwargs):
    """构造 JiT-L/32：24 blocks、1024 hidden、16 heads、32×32 patch。"""
    # 返回 JiT-L/32；256×256 输入时 token 数为 8×8=64。
    return JiT(depth=24, hidden_size=1024, num_heads=16,
               bottleneck_dim=128, in_context_len=32, in_context_start=8, patch_size=32, **kwargs)

def JiT_H_16(**kwargs):
    """构造 JiT-H/16：32 blocks、1280 hidden、16 heads、16×16 patch。"""
    # 返回 JiT-H/16；256×256 输入时 token 数为 16×16=256。
    return JiT(depth=32, hidden_size=1280, num_heads=16,
               bottleneck_dim=256, in_context_len=32, in_context_start=10, patch_size=16, **kwargs)

def JiT_H_32(**kwargs):
    """构造 JiT-H/32：32 blocks、1280 hidden、16 heads、32×32 patch。"""
    # 返回 JiT-H/32；256×256 输入时 token 数为 8×8=64。
    return JiT(depth=32, hidden_size=1280, num_heads=16,
               bottleneck_dim=256, in_context_len=32, in_context_start=10, patch_size=32, **kwargs)


# 注册所有可用模型名称到对应的构造函数；字典维度: 6 个名称 -> 6 个 callable。
JiT_models = {
    # JiT-B/16 构造器；配置: depth=12, hidden=768, heads=12, patch=16。
    'JiT-B/16': JiT_B_16,
    # JiT-B/32 构造器；配置: depth=12, hidden=768, heads=12, patch=32。
    'JiT-B/32': JiT_B_32,
    # JiT-L/16 构造器；配置: depth=24, hidden=1024, heads=16, patch=16。
    'JiT-L/16': JiT_L_16,
    # JiT-L/32 构造器；配置: depth=24, hidden=1024, heads=16, patch=32。
    'JiT-L/32': JiT_L_32,
    # JiT-H/16 构造器；配置: depth=32, hidden=1280, heads=16, patch=16。
    'JiT-H/16': JiT_H_16,
    # JiT-H/32 构造器；配置: depth=32, hidden=1280, heads=16, patch=32。
    'JiT-H/32': JiT_H_32,
}
