# --------------------------------------------------------
# References:
# SiT: https://github.com/willisma/SiT
# Lightning-DiT: https://github.com/hustvl/LightningDiT
# JiT: https://github.com/LTH14/JiT
# REPA: https://github.com/sihyun-yu/REPA
# RAE: https://github.com/bytetriper/RAE
# --------------------------------------------------------
import torch  # 张量运算。
import torch.nn as nn  # 可训练网络层。
import math  # 时间嵌入的指数频率。
import torch.nn.functional as F  # SDPA 注意力和 SiLU 激活。
from latent_force.model_util import VisionRotaryEmbeddingFast, get_2d_sincos_pos_embed, RMSNorm  # 二维 RoPE、固定位置编码、RMSNorm。

# 维度约定：B=batch，S=图像边长，p=pixel patch 边长，G=S/p，N=G²，C=hidden_size，D=DINO 通道数。
# 常用 S=256、p=16、G=16、N=256、D=768；C 随 S/B/L/H 模型配置变化。
# 本文件不运行预训练 DINO encoder；它接收 denoiser_cot.py 构造的两路带噪状态。
# 数据流：[B,3,S,S] / [B,D,G,G] -> 各自可训练 embedder -> 两个 [B,N,C] 逐元素相加。
# DINO 的输出 LayerNorm、数据集统计归一化、乘 0.485 均在 dinov2_hf.py，早于加扩散噪声和本文件的 embedder。
# 0.485 匹配的是干净 latent 与 [-1,1] pixel 的分量标准差，不保证投影后每个 token 的 L2 范数相等。
# 此处没有两路独立的投影后归一化/范数匹配；唯一序列 concat 是后续插入类别 in-context 前缀。
# 论文对照：Latent Forcing §4.1（全局方差匹配）、§4.2（两路 embedding 相加）。
# https://arxiv.org/html/2602.11401v1#S4


def modulate(x, shift, scale):  # x:[B,T,C]；条件生成的 shift/scale:[B,C]。
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)  # 在 token 轴广播，逐通道缩放和平移；形状不变。


class BottleneckPatchEmbed(nn.Module):  # 两层线性卷积的 patch 投影；pixel/DINO 分支各有独立参数。
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, pca_dim=768, embed_dim=768, bias=True):  # pca_dim 是可训练瓶颈宽度，并非执行 PCA。
        super().__init__()  # 初始化模块。
        img_size = (img_size, img_size)  # 输入空间尺寸 (H,W)。
        patch_size = (patch_size, patch_size)  # patch 尺寸/卷积步长。
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])  # 输出 token 数 N。
        self.img_size = img_size  # 保存用于输入尺寸检查。
        self.patch_size = patch_size  # 保存 patch 大小。
        self.num_patches = num_patches  # 保存序列长度。

        self.proj1 = nn.Conv2d(in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False)  # patch -> 瓶颈：[B,K,H/p,W/p]。
        self.proj2 = nn.Conv2d(pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias)  # 瓶颈 K -> hidden C；两层间没有激活/归一化。
        # pixel 和 DINO 分别经过各自的 BottleneckPatchEmbed。这个模块内部是两层卷积：第一层完成 patch 提取和 bottleneck 投影，
        # 第二层用 1x1 Conv 投影到 Transformer 的 hidden_size


    def forward(self, x):  # 输入通道优先的图像或 DINO 特征图。
        B, C, H, W = x.shape  # 此处 C 是输入通道数，不一定等于 Transformer hidden_size。
        assert H == self.img_size[0] and W == self.img_size[1], (  # 两个分支均要求配置的固定空间尺寸。
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."  # 报告实际尺寸及目标尺寸。
        )  # 尺寸不符时抛出断言错误。
        x = self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)  # [B,embed_dim,H/p,W/p] -> [B,N,embed_dim]。
        return x  # 不含位置编码，不做 token L2/RMS 归一化。

class NoBottleneckPatchEmbed(nn.Module):  # 无瓶颈备选模块；下方 JiTCoT 当前没有使用。
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, pca_dim=768, embed_dim=768, bias=True):  # pca_dim 保留在接口中，实际未使用。
        super().__init__()  # 初始化模块。
        img_size = (img_size, img_size)  # 输入 (H,W)。
        patch_size = (patch_size, patch_size)  # patch 的高、宽。
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])  # 输出 token 数。
        self.img_size = img_size  # 保存固定输入尺寸。
        self.patch_size = patch_size  # 保存 patch 大小。
        self.num_patches = num_patches  # 保存 token 数。

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)  # 单层 patch -> hidden 投影。

    def forward(self, x):  # x:[B,in_chans,H,W]。
        B, C, H, W = x.shape  # 解包输入形状。
        assert H == self.img_size[0] and W == self.img_size[1], (  # 检查空间尺寸。
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."  # 保留原始尺寸错误信息。
        )  # 尺寸不符时给出原始错误信息。
        x = self.proj(x).flatten(2).transpose(1, 2)  # [B,embed_dim,H/p,W/p] -> [B,N,embed_dim]。
        return x  # 未经过归一化的 token。


class TimestepEmbedder(nn.Module):  # 单路标量时间 -> hidden 条件向量；两路时间使用独立实例。
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):  # C=hidden_size，F=正弦频率嵌入宽度。
        super().__init__()  # 初始化模块。
        self.mlp = nn.Sequential(  # 对每个样本的时间向量进行非线性映射。
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),  # [B,F] -> [B,C]。
            nn.SiLU(),  # 逐元素激活。
            nn.Linear(hidden_size, hidden_size, bias=True),  # [B,C] -> [B,C]。
        )  # 时间 MLP。
        self.frequency_embedding_size = frequency_embedding_size  # 默认 F=256。

    @staticmethod  # 不依赖实例参数的固定频率编码。
    def timestep_embedding(t, dim, max_period=10000):  # t:[B]，dim=输出频率向量宽度。
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2  # cos/sin 各占一半通道。
        freqs = torch.exp(  # 对数间隔频率；无可训练参数。
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half  # [half]。
        ).to(device=t.device)  # 与输入时间在同一设备。
        args = t[:, None].float() * freqs[None]  # [B,half]；不在此处给 t 乘 1000。
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [B,2*half]；这是特征维拼接。
        if dim % 2:  # 奇数宽度补一个零通道。
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)  # [B,dim]。
        return embedding  # 固定正弦时间特征。

    def forward(self, t):  # t:[B]；训练调用通常是 [0,1] 连续时间。
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(t.dtype)  # [B,F]，恢复时间张量的数据类型。
        t_emb = self.mlp(t_freq)  # [B,C]。
        return t_emb  # 时间条件，不是额外的序列 token。


class LabelEmbedder(nn.Module):  # 类别 ID -> hidden 条件；实际 label dropout 在 denoiser_cot.py 中完成。
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size):  # 额外预留一个无条件类别 ID。
        super().__init__()  # 初始化模块。
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)  # 表形状 [K+1,C]；null ID=K。
        self.num_classes = num_classes  # 默认 K=1000。

    def forward(self, labels):  # labels:[B]，整数类别 ID。
        embeddings = self.embedding_table(labels)  # [B,C]，查询对应可训练类别向量。
        return embeddings  # 本函数本身不随机丢弃标签。


def scaled_dot_product_attention(query, key, value, dropout_p=0.0):  # q/k/v:[B,h,T,d]，T 含可能插入的类别前缀。
    return F.scaled_dot_product_attention(  # softmax(QKᵀ/sqrt(d))V，输出 [B,h,T,d]。
        query,  # 查询 Q。
        key,  # 键 K。
        value,  # 值 V。
        attn_mask=None,  # 无额外注意力屏蔽；不是 forward 的输入分支 mask。
        dropout_p=dropout_p,  # 训练时使用调用方提供的概率。
        is_causal=False  # 双向注意力，不是自回归因果注意力。
    )  # 具体高效实现由 PyTorch 选择。


class Attention(nn.Module):  # 融合后的单序列多头自注意力。
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0., proj_drop=0.):  # dim=C，num_heads=h。
        super().__init__()  # 初始化模块。
        self.num_heads = num_heads  # 注意力头数 h。
        head_dim = dim // num_heads  # 每头通道 d=C/h，要求整除。

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()  # 每个头的 Q 沿 d 做 RMSNorm。
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()  # 每个头的 K 沿 d 做 RMSNorm；V 不做。

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # 同时生成 Q/K/V。
        self.attn_drop = nn.Dropout(attn_drop)  # 此对象的 p 传入 SDPA，非单独调用。
        self.proj = nn.Linear(dim, dim)  # 多头合并后的输出投影。
        self.proj_drop = nn.Dropout(proj_drop)  # 输出 dropout。

    def forward(self, x, rope):  # x:[B,T,C]；rope 与当前 token 序列长度匹配。
        B, N, C = x.shape  # 此局部 N 是当前长度 T，可能为 256 或 288。
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # [3,B,h,T,d]。
        q, k, v = qkv[0], qkv[1], qkv[2]  # 各 [B,h,T,d]；显式索引兼容 TorchScript。

        q = self.q_norm(q)  # 归一化的是融合后 Q，不是原始 DINO/pixel 各自的 token。
        k = self.k_norm(k)  # K 同理；不是输入两路的范数匹配机制。

        q = rope(q)  # 对空间 token 加二维旋转位置编码；类别前缀不旋转。
        k = rope(k)  # 同一位置规则作用于 K，形状不变。

        x = scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)  # [B,h,T,d]；推理禁用 dropout。

        x = x.transpose(1, 2).reshape(B, N, C)  # 合并注意力头 -> [B,T,C]。

        x = self.proj(x)  # [B,T,C] -> [B,T,C]。
        x = self.proj_drop(x)  # 训练时随机丢弃输出分量。
        return x  # 残差分支的注意力输出。


class SwiGLUFFN(nn.Module):  # 带 SiLU 门控的前馈网络；保持 token 数和输出宽度。
    def __init__(  # 构造前馈投影。
        self,  # 当前实例。
        dim: int,  # 输入/输出通道 C。
        hidden_dim: int,  # 调整前的 FFN 中间宽度，通常 C*mlp_ratio。
        drop=0.0,  # 中间激活的 dropout 概率。
        bias=True  # 线性层是否使用偏置。
    ) -> None:  # 构造函数无返回值。
        super().__init__()  # 初始化模块。
        hidden_dim = int(hidden_dim * 2 / 3)  # 实际门控宽度 M，减少双分支带来的参数开销。
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)  # C -> 2M，一次生成两路激活。
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)  # M -> C。
        self.ffn_dropout = nn.Dropout(drop)  # 中间激活 dropout。

    def forward(self, x):  # x:[B,T,C]。
        x12 = self.w12(x)  # [B,T,2M]。
        x1, x2 = x12.chunk(2, dim=-1)  # 两个 [B,T,M]。
        hidden = F.silu(x1) * x2  # SiLU 门控乘法，仍为 [B,T,M]。
        return self.w3(self.ffn_dropout(hidden))  # [B,T,C]，交给外部残差连接。


class FinalLayer(nn.Module):  # 条件调制 + 逐 token 输出投影。
    """
    The final layer of JiT.
    """
    def __init__(self, hidden_size, out_channels, hidden_size_c=None):  # 输入宽度 Cx，输出宽度 O，条件宽度 Cc。
        super().__init__()  # 初始化模块。
        self.norm_final = RMSNorm(hidden_size)  # 沿每个 token 的 Cx 通道做 RMSNorm。
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)  # Cx -> O，O 是每个位置的预测分量数。
        self.adaLN_modulation = nn.Sequential(  # 名称沿用 adaLN，但基础归一化实际为 RMSNorm。
            nn.SiLU(),  # 对条件向量激活。
            nn.Linear(hidden_size_c or hidden_size, 2 * hidden_size, bias=True)  # [B,Cc] -> [B,2Cx]。
        )  # 生成平移和缩放，不生成门控。

    @torch.compile  # 编译此 forward 以优化执行。
    def forward(self, x, c):  # x:[B,N,Cx]，c:[B,Cc]。
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)  # 各 [B,Cx]。
        x = modulate(self.norm_final(x), shift, scale)  # 先 RMSNorm，再做条件仿射调制。
        x = self.linear(x)  # [B,N,O]，无 sigmoid/tanh，不限制预测值范围。
        return x  # 在 denoiser 中作为干净 pixel/latent 的 x-prediction 使用。


class JiTBlock(nn.Module):  # 条件控制的注意力/FFN 残差块；输入已是两路相加后的单序列。
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0, hidden_size_c=None):  # 可单独指定条件宽度 Cc。
        super().__init__()  # 初始化模块。
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)  # 注意力前 RMSNorm；不是融合前的两路独立归一化。
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,  # 带 Q/K RMSNorm 的自注意力。
                              attn_drop=attn_drop, proj_drop=proj_drop)  # 传入两种 dropout 概率。
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)  # FFN 前的 RMSNorm。
        mlp_hidden_dim = int(hidden_size * mlp_ratio)  # SwiGLU 调整前的中间宽度。
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)  # 前馈输出保持 hidden_size。
        self.adaLN_modulation = nn.Sequential(  # 条件向量生成两条残差分支的仿射参数和门控。
            nn.SiLU(),  # 激活条件。
            nn.Linear(hidden_size_c or hidden_size, 6 * hidden_size, bias=True)  # [B,Cc] -> [B,6C]。
        )  # 每条分支各有 shift、scale、gate。

    @torch.compile  # 编译残差块 forward。
    def forward(self, x,  c, feat_rope=None):  # x:[B,T,C]，c:[B,Cc]，RoPE 由调用方提供。
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)  # 六个 [B,C]。
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)  # 门控注意力残差；形状不变。
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))  # 门控 FFN 残差；形状不变。
        return x  # [B,T,C]。


class JiTCoT(nn.Module):  # 联合去噪 pixel/DINO 的共享 Transformer；不是预训练 DINO encoder。
    """
    Just image Transformer.
    """
    def __init__(  # 默认参数对应 L/16 主干；工厂函数可覆盖。
        self,  # 当前联合去噪模型。
        input_size=256,  # pixel 输入边长 S。
        patch_size=16,  # pixel patch 边长 p；DINO embedder 的 patch 固定为 1。
        in_channels=3,  # pixel 输入/预测通道数，通常 RGB=3。
        hidden_size=1024,  # 共享主干 token 宽度 C。
        depth=24,  # 共享 Transformer block 数。
        num_heads=16,  # 共享主干注意力头数。
        mlp_ratio=4.0,  # FFN 基础扩展比例，SwiGLU 内另乘 2/3。
        attn_drop=0.0,  # 注意力权重 dropout。
        proj_drop=0.0,  # 注意力输出/FFN 中间激活 dropout。
        num_classes=1000,  # 有条件类别数，另有一个 null 类。
        bottleneck_dim=128,  # pixel embedder 瓶颈宽度 Kp。
        bottleneck_dim_dino=128,  # DINO embedder 瓶颈宽度 Kd。
        in_context_len=32,  # 类别前缀长度 L；不是 DINO 特征 token 数。
        in_context_start=8,  # 在第几个 block 前插入前缀，按 0 开始计数。
        dino_in_channels=768,  # DINO 特征通道 D；默认 DINOv2-B 为 768。
        dh_depth=0,  # >0 时增加两路独立去噪 head 的 Transformer blocks。
        dh_hidden_size=1536,  # 可选独立 head 的 token 宽度 Cdh。
    ):  # 构造主干、嵌入器和预测头。
        super().__init__()  # 初始化模块。
        self.in_channels = in_channels  # pixel 通道数。
        self.out_channels_pixel = in_channels * patch_size * patch_size  # 每个位置预测一个 pixel patch：Op=3p²，p=16 时为 768。
        self.out_channels_dino = dino_in_channels  # 每个位置预测一个 DINO 向量：Od=D=768。
        self.out_channels = self.out_channels_pixel + self.out_channels_dino  # 单头总输出宽度 Op+Od；不是输入 token 拼接。
        self.patch_size = patch_size  # pixel unpatchify 使用的 p。
        self.num_heads = num_heads  # 主干注意力头数。
        self.hidden_size = hidden_size  # 主干宽度 C。
        self.input_size = input_size  # pixel 空间尺寸 S。
        self.in_context_len = in_context_len  # 类别前缀长度 L。
        self.in_context_start = in_context_start  # 前缀插入层索引。
        self.num_classes = num_classes  # 类别数 K。

        # time and class embed
        self.t_pixel_embedder = TimestepEmbedder(hidden_size)  # pixel 噪声时间 t_pixel -> [B,C]。
        self.t_dino_embedder = TimestepEmbedder(hidden_size)  # DINO 噪声时间 t_dino -> [B,C]；与上行不共享权重。
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)  # 类别 ID -> [B,C]。

        # linear embed
        self.pixel_embedder = BottleneckPatchEmbed(input_size, patch_size, in_channels, bottleneck_dim, hidden_size, bias=True)  # [B,3,S,S] -> [B,Kp,G,G] -> [B,N,C]。
        self.dino_embedder = BottleneckPatchEmbed(input_size//patch_size, 1, dino_in_channels, bottleneck_dim_dino, hidden_size, bias=True)  # [B,D,G,G] -> [B,Kd,G,G] -> [B,N,C]；可训练适配器，不是 DINO encoder。

        # use fixed sin-cos embedding
        num_patches = self.pixel_embedder.num_patches  # N=(S/p)²；默认 256，两个分支的网格必须一致。
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)  # [1,N,C]，稍后填入固定二维 sin/cos。

        # in-context cls token
        if self.in_context_len > 0:  # 启用类别前缀时才创建其位置参数。
            self.in_context_posemb = nn.Parameter(torch.zeros(1, self.in_context_len, hidden_size), requires_grad=True)  # [1,L,C]，各前缀位置可训练。
            torch.nn.init.normal_(self.in_context_posemb, std=.02)  # 以小幅随机值区分重复的类别 token。

        # rope
        half_head_dim = hidden_size // num_heads // 2  # 每头维度的一半；二维 RoPE 的高、宽方向各占一半。
        hw_seq_len = input_size // patch_size  # 空间 token 网格边长 G。
        self.feat_rope = VisionRotaryEmbeddingFast(  # 无类别前缀的 RoPE，长度 N。
            dim=half_head_dim,  # 单个空间轴编码宽度。
            pt_seq_len=hw_seq_len,  # 二维网格边长。
            num_cls_token=0  # 不额外预留前缀位置。
        )  # 用于前缀插入前的主干，以及独立预测头。
        self.feat_rope_incontext = VisionRotaryEmbeddingFast(  # 带前缀的 RoPE，长度 L+N。
            dim=half_head_dim,  # 与上面的每头旋转宽度一致。
            pt_seq_len=hw_seq_len,  # 图像 token 的二维网格不变。
            num_cls_token=self.in_context_len  # 前 L 个位置 cos=1/sin=0，即类别前缀不旋转。
        )  # 用于前缀插入后的主干层。

        # transformer
        self.blocks = nn.ModuleList([  # 共享主干依次处理已融合的 token。
            JiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,  # 所有层维持主干宽度 C。
                     attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,  # 仅中间指定层段启用注意力 dropout。
                     proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0)  # 输出/FFN dropout 使用相同层段。
            for i in range(depth)  # block 索引从 0 到 depth-1。
        ])  # 注册所有主干 block 参数。

        # linear predict
        self.dh_depth = dh_depth  # 保存独立去噪 head 深度。
        if self.dh_depth > 0:  # 两个独立 head 都接收同一份主干特征。
            self.dh_pixel_proj = nn.Linear(hidden_size, dh_hidden_size)  # pixel head：[B,N,C] -> [B,N,Cdh]。
            self.dh_dino_proj = nn.Linear(hidden_size, dh_hidden_size)  # DINO head：独立参数，形状相同。

            self.dh_blocks_pixel = nn.ModuleList([  # pixel 专用 Transformer blocks。
                JiTBlock(dh_hidden_size, num_heads * dh_hidden_size // hidden_size , mlp_ratio=mlp_ratio, hidden_size_c=hidden_size) for _ in range(dh_depth)  # 条件宽度仍为 C；整除配置下保持每头维度。
            ])  # 与 DINO head 不共享参数。
            self.dh_blocks_dino = nn.ModuleList([  # DINO 专用 Transformer blocks。
                JiTBlock(dh_hidden_size, num_heads * dh_hidden_size // hidden_size , mlp_ratio=mlp_ratio, hidden_size_c=hidden_size) for _ in range(dh_depth)  # 接收同一条件 c:[B,C]。
            ])  # head 中不再插入类别前缀。

            self.final_layer_pixel = FinalLayer(dh_hidden_size, self.out_channels_pixel, hidden_size_c=hidden_size)  # [B,N,Cdh] -> [B,N,Op]。
            self.final_layer_dino = FinalLayer(dh_hidden_size, self.out_channels_dino, hidden_size_c=hidden_size)  # [B,N,Cdh] -> [B,N,Od]。
        else:  # 不使用独立 Transformer head。
            self.final_layer = FinalLayer(hidden_size, self.out_channels)  # 一次投影为 [B,N,Op+Od]，forward 再按通道拆开。

        self.initialize_weights()  # 初始化可训练去噪网络；与 dinov2_hf.py 加载预训练 encoder 无关。

    def initialize_weights(self):  # 初始化 JiT 去噪网络；此处没有任何预训练 DINO 权重加载。
        # Initialize transformer layers:
        def _basic_init(module):  # 统一初始化所有线性层，特例随后覆盖。
            if isinstance(module, nn.Linear):  # 不包括 Conv2d 和 Embedding。
                torch.nn.init.xavier_uniform_(module.weight)  # 按输入/输出宽度设置随机权重尺度。
                if module.bias is not None:  # 仅处理存在的偏置。
                    nn.init.constant_(module.bias, 0)  # 线性层偏置置零。
        self.apply(_basic_init)  # 递归遍历全部子模块。

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.pixel_embedder.num_patches ** 0.5))  # 构造 [N,C] 的二维 sin/cos。
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))  # 写入 [1,N,C]；requires_grad=False，训练中固定。

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w1 = self.pixel_embedder.proj1.weight.data  # pixel 第一层卷积权重 [Kp,3,p,p]。
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))  # 当作 [Kp,3p²] 的 patch 线性投影初始化。
        w2 = self.pixel_embedder.proj2.weight.data  # pixel 第二层权重 [C,Kp,1,1]。
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))  # 当作 [C,Kp] 初始化。
        nn.init.constant_(self.pixel_embedder.proj2.bias, 0)  # 第二层偏置置零；第一层没有偏置。

        w1 = self.dino_embedder.proj1.weight.data  # DINO 适配器第一层 [Kd,D,1,1]；不是预训练 encoder。
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))  # 当作 [Kd,D] 随机初始化，并在去噪训练中更新。
        w2 = self.dino_embedder.proj2.weight.data  # DINO 适配器第二层 [C,Kd,1,1]。
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))  # 当作 [C,Kd] 初始化；不共享 pixel 分支权重。
        nn.init.constant_(self.dino_embedder.proj2.bias, 0)  # 第二层偏置置零；没有额外 norm/gate。

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)  # 包括 null 类的 [K+1,C] 类别向量。

        nn.init.normal_(self.t_pixel_embedder.mlp[0].weight, std=0.02)  # pixel 时间 MLP 第一层。
        nn.init.normal_(self.t_pixel_embedder.mlp[2].weight, std=0.02)  # pixel 时间 MLP 第二层。
        nn.init.normal_(self.t_dino_embedder.mlp[0].weight, std=0.02)  # DINO 时间 MLP 第一层，独立采样。
        nn.init.normal_(self.t_dino_embedder.mlp[2].weight, std=0.02)  # DINO 时间 MLP 第二层。

        # Zero-out adaLN modulation layers:
        for block in self.blocks:  # 共享主干各层。
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)  # 六组调制参数的权重置零。
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)  # 初始 gate=0，因此各残差块初始为恒等映射。

        if self.dh_depth > 0:  # 存在独立 head 时，使用相同的零门控初始化。
            for block in self.dh_blocks_pixel:  # pixel head 的各层。
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)  # 调制权重置零。
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)  # 初始为恒等残差块。
            for block in self.dh_blocks_dino:  # DINO head 的各层。
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)  # 调制权重置零。
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)  # 初始为恒等残差块。

            # w1 = self.pixel_embedder_dh.proj.weight.data
            # nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
            # nn.init.constant_(self.pixel_embedder_dh.proj.bias, 0)
            # w1 = self.dino_embedder_dh.proj.weight.data
            # nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
            # nn.init.constant_(self.dino_embedder_dh.proj.bias, 0)

        # Zero-out output layers:
        for final_layer in [self.final_layer] if self.dh_depth == 0 else [self.final_layer_pixel, self.final_layer_dino]:  # 遍历实际使用的输出头。
            nn.init.constant_(final_layer.adaLN_modulation[-1].weight, 0)  # 初始 shift/scale 不依赖条件。
            nn.init.constant_(final_layer.adaLN_modulation[-1].bias, 0)  # 初始 shift=scale=0。
            nn.init.constant_(final_layer.linear.weight, 0)  # 最终输出权重置零。
            nn.init.constant_(final_layer.linear.bias, 0)  # 初始化时 pixel/DINO 预测都为零。

    def unpatchify(self, x, p, c):  # x:[B,N,p²c]；下方原 docstring 的 imgs 排列不准确，实际返回 NCHW。
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        h = w = int(x.shape[1] ** 0.5)  # token 网格边长，默认 16。
        assert h * w == x.shape[1]  # 要求 token 数构成正方形。

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))  # [B,h,w,p,p,c]，拆出每个 patch 内的像素。
        x = torch.einsum('nhwpqc->nchpwq', x)  # [B,c,h,p,w,p]，将 patch 内外空间轴交错排列。
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))  # [B,c,hp,wp]；此处 h=w。
        return imgs  # pixel 用 p=16/c=3；DINO 用 p=1/c=D，仅恢复特征图，不经过 decoder。

    def forward(self, x, t, y, m=None):  # x 为两路带噪状态的 list；本函数不调用 DINO encoder。
        """
        x: (N, C, H, W), (N D HP WP)
        t: (N,), (N,)
        y: (N,)
        m: Optional (N, 256), (N, 256)
        """
        # 时间/类别条件路径：相加成一个 [B,C] 向量，区别于下面的空间 token 融合。
        t_emb_pixel = self.t_pixel_embedder(t[0])  # t[0]:[B] -> [B,C]，pixel 分支的扩散时间。
        t_emb_dino = self.t_dino_embedder(t[1])  # t[1]:[B] -> [B,C]，DINO 分支时间可与 pixel 不同。
        y_emb = self.y_embedder(y)  # y:[B] -> [B,C]，含用于 CFG 的 null 类。
        c = t_emb_pixel + t_emb_dino + y_emb  # [B,C]，传给每层调制器；原 TODO：尝试改为条件 tokens？

        # 空间状态路径：latent 的尺度处理已在 RAE.encode 中完成，然后由 denoiser 分别加噪。
        x_pixel, x_dino = x  # 默认 [B,3,256,256] 和 [B,768,16,16]；对应同一组 N=256 空间位置。
        if m is None:  # 不使用模态 token mask。
            x = self.pixel_embedder(x_pixel) + self.dino_embedder(x_dino)  # 两个 [B,N,C] 逐元素相加，仍为 [B,N,C]；不是 [B,2N,C] concat。
        else:  # m[0]/m[1] 为各分支 [B,N] bool mask，True 表示屏蔽。
            x = self.pixel_embedder(x_pixel) * (~m[0][...,None]) + self.dino_embedder(x_dino) * (~m[1][...,None])  # 各自投影后按 token 清零，再相加；无额外 norm/融合门控。
        x += self.pos_embed  # 加一次共享 [1,N,C] 空间位置编码；不是两路各加后再融合。

        for i, block in enumerate(self.blocks):  # 共享主干逐层处理融合序列。
            # 类别 in-context 前缀：这是 torch.cat 的位置，但拼接对象不是 pixel/DINO 两路。
            if self.in_context_len > 0 and i == self.in_context_start:  # 在指定层前插入一次。
                in_context_tokens = y_emb.unsqueeze(1).repeat(1, self.in_context_len, 1)  # 类别向量复制 L 次 -> [B,L,C]。
                in_context_tokens += self.in_context_posemb  # 加 [1,L,C] 可训练前缀位置，使各副本可区分。
                x = torch.cat([in_context_tokens, x], dim=1)  # 沿序列维：[B,L,C]+[B,N,C] -> [B,L+N,C]，默认 288 个 token。
            x = block(x, c, self.feat_rope if i < self.in_context_start else self.feat_rope_incontext)  # 插入前后切换对应 RoPE；block 内 RMSNorm 作用于已融合特征。

        x = x[:, self.in_context_len:]  # 去掉类别前缀 -> [B,N,C]；假定启用前缀时插入层在主干范围内。

        if self.dh_depth == 0:  # 共享最终预测层的配置。
            output_pixel, output_dino = self.final_layer(x, c).split([self.out_channels_pixel, self.out_channels_dino], dim=-1)  # [B,N,Op+Od] 按特征维拆为 [B,N,Op]/[B,N,Od]。
        else:  # 具有两路独立 Transformer 预测 head 的配置。
            dh_feats_pixel = self.dh_pixel_proj(x)  # [B,N,C] -> [B,N,Cdh]，pixel head 输入。
            dh_feats_dino = self.dh_dino_proj(x)  # 同一主干特征经另一投影，进入 DINO head。

            for i, block in enumerate(self.dh_blocks_pixel):  # pixel 专用 blocks，不再含类别前缀。
                dh_feats_pixel = block(dh_feats_pixel, c, self.feat_rope)  # [B,N,Cdh]；条件 c 仍为 [B,C]。
            for i, block in enumerate(self.dh_blocks_dino):  # DINO 专用 blocks，与 pixel 不共享参数。
                dh_feats_dino = block(dh_feats_dino, c, self.feat_rope)  # [B,N,Cdh]；保持相同空间网格。

            output_pixel = self.final_layer_pixel(dh_feats_pixel, c)  # [B,N,3p²]，预测干净 pixel patch。
            output_dino = self.final_layer_dino(dh_feats_dino, c)  # [B,N,D]，预测尺度处理后的干净 latent。

        output_pixel = self.unpatchify(output_pixel, self.patch_size, self.in_channels)  # 默认 [B,256,768] -> [B,3,256,256]。
        output_dino = self.unpatchify(output_dino, 1, self.out_channels_dino)  # 默认 [B,256,768] -> [B,768,16,16]。
        output = [output_pixel, output_dino]  # 两路预测分别返回，不把输出相加，也不调用 RAE decoder。

        return output  # 交由 denoiser 计算两路损失或采样更新。


def JiTCoT_S_16(**kwargs):  # S/16：主干 12 层、宽度 384、6 头。
    return JiTCoT(depth=12, hidden_size=384, num_heads=6,  # 每头 64 维；默认 pixel/DINO 均为 256 个 token。
               bottleneck_dim=128, in_context_len=32, in_context_start=4, patch_size=16, **kwargs)  # 第 4 层前插入 32 个类别 token。


def JiTCoT_B_16(**kwargs):  # B/16：主干 12 层、宽度 768、12 头。
    return JiTCoT(depth=12, hidden_size=768, num_heads=12,  # 每头 64 维。
               bottleneck_dim=128, in_context_len=32, in_context_start=4, patch_size=16, **kwargs)  # pixel 瓶颈 128；DINO 瓶颈沿用默认 128。

def JiTCoT_B_32(**kwargs):  # B/32：N=(S/32)²；须同步保证外部 DINO 网格与其一致。
    return JiTCoT(depth=12, hidden_size=768, num_heads=12,  # 同 B/16 主干，pixel patch 更大。
               bottleneck_dim=128, in_context_len=32, in_context_start=4, patch_size=32, **kwargs)  # S=256 时要求 8×8 latent，与 RAE 默认 16×16 不直接匹配。

def JiTCoT_L_16(**kwargs):  # L/16：主干 24 层、宽度 1024、16 头。
    return JiTCoT(depth=24, hidden_size=1024, num_heads=16,  # 每头 64 维。
               bottleneck_dim=128, in_context_len=32, in_context_start=8, patch_size=16, **kwargs)  # 第 8 层前插入类别前缀。

def JiTCoT_LM_16(**kwargs):  # LM/16：20 层、宽度 1024；不是 24 层 L 配置。
    return JiTCoT(depth=20, hidden_size=1024, num_heads=16,  # 每头 64 维。
               bottleneck_dim=128, in_context_len=32, in_context_start=8, patch_size=16, **kwargs)  # 独立 head 深度可通过 kwargs 指定。

def JiTCoT_L_32(**kwargs):  # L/32：主干同 L/16，N=(S/32)²。
    return JiTCoT(depth=24, hidden_size=1024, num_heads=16,  # 宽度 1024，16 头。
               bottleneck_dim=128, in_context_len=32, in_context_start=8, patch_size=32, **kwargs)  # 同样要求外部 latent 网格匹配 S/32。

def JiTCoT_H_16(**kwargs):  # H/16：主干 32 层、宽度 1280、16 头，每头 80 维。
    return JiTCoT(depth=32, hidden_size=1280, num_heads=16,  # 大主干配置。
               bottleneck_dim=256, in_context_len=32, in_context_start=10, patch_size=16, **kwargs)  # pixel 瓶颈变 256；DINO 瓶颈仍默认 128。

def JiTCoT_H_32(**kwargs):  # H/32：同 H/16 主干，使用 32×32 pixel patch。
    return JiTCoT(depth=32, hidden_size=1280, num_heads=16,  # 每头 80 维。
               bottleneck_dim=256, in_context_len=32, in_context_start=10, patch_size=32, **kwargs)  # 需对应 S/32 的 latent 网格。


JiTCoT_models = {  # CLI 模型名称 -> 构造函数。
    'JiTCoT-S/16': JiTCoT_S_16,  # Small，patch=16。
    'JiTCoT-B/16': JiTCoT_B_16,  # Base，patch=16。
    'JiTCoT-B/32': JiTCoT_B_32,  # Base，patch=32。
    'JiTCoT-L/16': JiTCoT_L_16,  # Large，24 层，patch=16。
    'JiTCoT-LM/16': JiTCoT_LM_16,  # 20 层的 Large 变体，patch=16。
    'JiTCoT-L/32': JiTCoT_L_32,  # Large，patch=32。
    'JiTCoT-H/16': JiTCoT_H_16,  # Huge，patch=16。
    'JiTCoT-H/32': JiTCoT_H_32,  # Huge，patch=32。
}  # denoiser_cot.py 通过 args.model 在此查找模型。

# TODO halve heads
