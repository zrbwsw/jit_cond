"""最小 Latent Forcing 子集：只用其 latent 阶段生成 grounding token。

只保留从 `LatentForcing/` 搬运过来的 ``model_cot.py``（JiTCoT 联合去噪网络）
与 ``model_util.py``（RoPE/RMSNorm/固定位置编码）。原项目中的 RAE/DINOv2 编码器
（``dinov2_hf.py``）、pixel 阶段、训练逻辑、EMA 维护都没有搬入：本目录只在采样
时调用预训练权重前 25 步的 latent 轨迹，产出 ``[B,768,16,16]`` 的 DINOv2 特征图。

时间约定与 JiT/Latent Forcing 一致：``t=0`` 为纯噪声，``t=1`` 为干净端点。
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

from latent_force.model_cot import JiTCoT_models


# lfjit-l-200.pth 的实际结构（从 checkpoint 张量形状反推）：
# 主干 20 层、宽度 1024、16 头；两个独立输出 head 各 2 层、宽度 1024；
# DINO 输入通道 768；类别数 1000（另有 null 类）。
LF_MODEL_NAME = "JiTCoT-LM/16"
LF_MODEL_KWARGS = dict(
    input_size=256,
    in_channels=3,
    num_classes=1000,
    bottleneck_dim_dino=128,
    dino_in_channels=768,
    dh_depth=2,
    dh_hidden_size=1024,
    attn_drop=0.0,
    proj_drop=0.0,
)


def build_ji_tcot(**overrides):
    """构造与 ``lfjit-l-200.pth`` 匹配的 JiTCoT 网络。"""
    kwargs = dict(LF_MODEL_KWARGS)
    kwargs.update(overrides)
    return JiTCoT_models[LF_MODEL_NAME](**kwargs)


def load_lf_net(ckpt: str, ema: str = "1", device="cuda", dtype=torch.float32):
    """加载冻结的 Latent Forcing 网络。

    checkpoint 是 ``DenoiserCoT`` 的 state_dict，含 ``net.*``（联合去噪网络）与
    ``dino.*``（冻结的 DINOv2 编码器）。latent 采样从纯噪声开始，不调用编码器，
    因此只取 ``net.*``，DINOv2 那部分不加载，也就不需要它的权重文件和 RAE 统计量。
    """
    state = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
    key = {"1": "model_ema1", "2": "model_ema2", "none": "model"}[str(ema)]
    if key not in state:
        raise KeyError(f"checkpoint {ckpt} 中没有 {key}；实际包含 {sorted(state.keys())}")

    net = build_ji_tcot()
    net_state = {k[len("net."):]: v for k, v in state[key].items() if k.startswith("net.")}
    if not net_state:
        raise KeyError(f"checkpoint {ckpt}[{key}] 中没有 net.* 权重")
    missing, unexpected = net.load_state_dict(net_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "JiTCoT 结构与 checkpoint 不匹配："
            f"missing={list(missing)[:8]} unexpected={list(unexpected)[:8]}"
        )

    net = net.to(device=device, dtype=dtype)
    net.eval()
    net.requires_grad_(False)
    return net


class LatentForcingSampler:
    """冻结的 Latent Forcing 网络 + 只推进 latent 的 25 步采样。

    对应 ``LatentForcing/denoiser_cot.py`` 中 ``sample_mode=dino_first_cascaded*``
    的级联调度：前一半步数把 latent 从纯噪声推到 ``dino_max_t``，此时 pixel 分支
    始终停在 ``t=0``（纯噪声）。本类只跑这一段，最后返回 latent 状态。

    与原始实现保持一致的部分：
    * 时间网格 ``linspace(0, dino_max_t, M+1)`` 再经 shift 映射
      ``f_s(t) = s*t / (1 + (s-1)*t)``；
    * 每步用共享网络同时预测两路，按各自 ``Δt`` 积分（pixel 的 ``Δt=0``，状态不变）；
    * guidance 为 ``v_ref + scale*(v_cond - v_ref)``，区间按该模态自身时间判定。
    """

    def __init__(
        self,
        ckpt: str,
        latent_steps: int = 25,
        sampling_method: str = "heun",
        dino_max_t: float = 1.0,
        gen_shift_dino: float = 1.0,
        gen_shift_pixel: float = 1.0,
        cfg_scale: float = 1.0,
        cfg_interval: Sequence[float] = (0.0, 1.0),
        guidance_method: str = "none",
        num_classes: int = 1000,
        latent_channels: int = 768,
        latent_size: int = 16,
        img_size: int = 256,
        noise_scale: float = 1.0,
        t_eps: float = 0.05,
        ema: str = "1",
        device="cuda",
        dtype=torch.float32,
    ):
        if sampling_method not in ("euler", "heun"):
            raise ValueError(f"不支持的 sampling_method: {sampling_method}")
        if guidance_method not in ("none", "cfg"):
            raise ValueError(f"不支持的 guidance_method: {guidance_method}")
        if latent_steps < 1:
            raise ValueError("latent_steps 必须为正整数")

        self.net = load_lf_net(ckpt, ema=ema, device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype

        self.latent_steps = int(latent_steps)
        self.method = sampling_method
        self.dino_max_t = float(dino_max_t)
        self.gen_shift_dino = float(gen_shift_dino)
        self.gen_shift_pixel = float(gen_shift_pixel)
        self.cfg_scale = float(cfg_scale)
        self.cfg_interval = (float(cfg_interval[0]), float(cfg_interval[1]))
        self.guidance_method = guidance_method

        self.num_classes = int(num_classes)
        self.latent_channels = int(latent_channels)
        self.latent_size = int(latent_size)
        self.img_size = int(img_size)
        self.noise_scale = float(noise_scale)
        self.t_eps = float(t_eps)

    # ---------------------------------------------------------------- 调度

    @staticmethod
    def _shift(t: torch.Tensor, s: float) -> torch.Tensor:
        """时间映射 ``f_s(t) = s*t / (1 + (s-1)*t)``；``s=1`` 时保持恒等。"""
        return t * s / (1.0 + (s - 1.0) * t)

    def latent_grid(self, device=None) -> torch.Tensor:
        """latent 阶段的 ``M+1`` 个时间节点，与 LF 级联调度的前半段一致。"""
        device = self.device if device is None else device
        y = torch.linspace(0.0, self.dino_max_t, self.latent_steps + 1, device=device)
        return self._shift(y, self.gen_shift_dino)

    # ---------------------------------------------------------------- 网络

    def _velocity(self, z: List[torch.Tensor], t: Sequence[torch.Tensor], labels: torch.Tensor):
        """返回两路 guidance 速度；``z=[z_pixel, z_dino]``、``t=[t_pixel, t_dino]``。"""
        t_flat = [t[0].flatten(), t[1].flatten()]
        x_cond = self.net(z, t_flat, labels)
        v_cond = [(x_cond[i] - z[i]) / (1.0 - t[i]).clamp_min(self.t_eps) for i in (0, 1)]

        if self.guidance_method == "none" or self.cfg_scale == 1.0:
            return v_cond

        # 无条件分支：同一网络 + null 类（LF 训练时 label_drop_prob=0.1）。
        null_labels = torch.full_like(labels, self.num_classes)
        x_ref = self.net(z, t_flat, null_labels)
        v_ref = [(x_ref[i] - z[i]) / (1.0 - t[i]).clamp_min(self.t_eps) for i in (0, 1)]

        low, high = self.cfg_interval
        guided = []
        for i in (0, 1):
            mask = (t[i] < high) & ((low == 0) | (t[i] > low))
            scale = torch.where(mask, self.cfg_scale, 1.0)
            guided.append(v_ref[i] + scale * (v_cond[i] - v_ref[i]))
        return guided

    def _euler(self, z, t, t_next, labels):
        v = self._velocity(z, t, labels)
        return [z[i] + (t_next[i] - t[i]) * v[i] for i in (0, 1)]

    def _heun(self, z, t, t_next, labels):
        v_t = self._velocity(z, t, labels)
        z_euler = [z[i] + (t_next[i] - t[i]) * v_t[i] for i in (0, 1)]
        v_next = self._velocity(z_euler, t_next, labels)
        v = [0.5 * (v_t[i] + v_next[i]) for i in (0, 1)]
        return [z[i] + (t_next[i] - t[i]) * v[i] for i in (0, 1)]

    # ---------------------------------------------------------------- 采样

    @torch.no_grad()
    def sample_latent(self, labels: torch.Tensor, chunk_size: Optional[int] = None) -> torch.Tensor:
        """从纯噪声采样 latent；返回 ``[B,768,16,16]``（LF 的 DINOv2 特征空间）。

        ``chunk_size`` 只用于控制显存，分块结果与整批采样等价（每块使用独立噪声）。
        """
        if labels.ndim != 1:
            raise ValueError(f"labels 必须是 [B]，实际 {tuple(labels.shape)}")
        if chunk_size is None or chunk_size >= labels.shape[0]:
            return self._sample_chunk(labels)
        return torch.cat(
            [self._sample_chunk(labels[i:i + chunk_size])
             for i in range(0, labels.shape[0], chunk_size)],
            dim=0,
        )

    @torch.no_grad()
    def _sample_chunk(self, labels: torch.Tensor) -> torch.Tensor:
        bsz = labels.shape[0]
        dev = labels.device
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev.type == "cuda")):
            z_pixel = self.noise_scale * torch.randn(
                bsz, 3, self.img_size, self.img_size, device=dev, dtype=self.dtype)
            z_dino = self.noise_scale * torch.randn(
                bsz, self.latent_channels, self.latent_size, self.latent_size,
                device=dev, dtype=self.dtype)
            z = [z_pixel, z_dino]

            # pixel 分支在 latent 阶段始终保持在 t=0（纯噪声），Δt=0 使其状态不变。
            t_pixel = torch.zeros(bsz, 1, 1, 1, device=dev, dtype=self.dtype)
            grid = self.latent_grid(dev)

            stepper = self._heun if self.method == "heun" else self._euler
            for i in range(self.latent_steps):
                t = [t_pixel, grid[i].expand(bsz).view(-1, 1, 1, 1)]
                t_next = [t_pixel, grid[i + 1].expand(bsz).view(-1, 1, 1, 1)]
                z = stepper(z, t, t_next, labels)

        return z[1].float()


class LatentForcingGrounding:
    """采样时的 grounding 来源：用 Latent Forcing 生成 DINOv2 语义特征。

    返回 ``[B,N,768]``（``N=256``），形状与 ``DINOv3GroundingEncoder`` 的
    clean-image patch token 完全一致，因此可以直接喂给 JiT 的 patch-level AdaLN。
    """

    def __init__(self, args):
        self.sampler = LatentForcingSampler(
            ckpt=args.lf_ckpt,
            latent_steps=getattr(args, "lf_latent_steps", 25),
            sampling_method=getattr(args, "lf_sampling_method", "heun"),
            dino_max_t=getattr(args, "lf_dino_max_t", 1.0),
            gen_shift_dino=getattr(args, "lf_gen_shift_dino", 1.0),
            gen_shift_pixel=getattr(args, "lf_gen_shift_pixel", 1.0),
            cfg_scale=getattr(args, "lf_cfg", 1.0),
            cfg_interval=(getattr(args, "lf_interval_min", 0.0),
                          getattr(args, "lf_interval_max", 1.0)),
            guidance_method=getattr(args, "lf_guidance", "none"),
            num_classes=getattr(args, "class_num", 1000),
            ema=str(getattr(args, "lf_ema", "1")),
            device=getattr(args, "device", "cuda"),
        )
        self.chunk_size = getattr(args, "lf_gen_bsz", None) or None

    @torch.no_grad()
    def __call__(self, labels: torch.Tensor) -> torch.Tensor:
        latent = self.sampler.sample_latent(labels, chunk_size=self.chunk_size)
        # [B,768,16,16] -> [B,256,768]，与 clean patch token 的布局一致。
        return latent.flatten(2).transpose(1, 2).contiguous()
