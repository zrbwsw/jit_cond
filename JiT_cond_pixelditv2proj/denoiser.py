"""JiT flow matching with noisy-state DINO grounding and a time-conditioned projector."""

import torch
import torch.nn as nn

from model_jit import JiT_models
from grounding import DINOv3GroundingEncoder


class Denoiser(nn.Module):
    def __init__(self, args):
        """构造 JiT、可训练的时间 projector 和冻结的 DINO；t=0 为噪声，t=1 为干净图。"""
        super().__init__()
        self.use_grounding = bool(getattr(args, "use_grounding", getattr(args, "enable_grounding", True)))
        self.net = JiT_models[args.model](
            input_size=args.img_size,
            in_channels=3,
            num_classes=args.class_num,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
            grounding_dim=(getattr(args, "grounding_dim", 768) if self.use_grounding else None),
            grounding_proj_blocks=getattr(args, "grounding_proj_blocks", 1),
            grounding_proj_heads=getattr(args, "grounding_proj_heads", None),
        )
        self.img_size = args.img_size
        self.num_classes = args.class_num
        self.label_drop_prob = args.label_drop_prob
        self.ground_drop_prob = float(getattr(args, "ground_drop_prob", 0.0))
        if not 0 <= self.label_drop_prob <= 1 or not 0 <= self.ground_drop_prob <= 1:
            raise ValueError("label_drop_prob and ground_drop_prob must lie in [0, 1]")
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale

        if self.use_grounding:
            # DINO 输入网格与 JiT patch 网格一致；B/16@256 为 [B,256,768]。
            self.grounding_encoder = DINOv3GroundingEncoder(
                model_name=getattr(args, "grounding_model", "dinov3_vitb16"),
                weights_path=getattr(args, "grounding_weights", None) or None,
                repo=getattr(args, "grounding_repo", None) or None,
                target_patch_size=self.net.patch_size,
            )
            if self.grounding_encoder.embed_dim != self.net.grounding_dim:
                raise ValueError(
                    f"DINO feature width {self.grounding_encoder.embed_dim} does not match "
                    f"grounding_dim={self.net.grounding_dim}"
                )
            self.grounding_encoder.requires_grad_(False)
            self.grounding_encoder.eval()
        else:
            self.grounding_encoder = None

        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        if self.steps < 1:
            raise ValueError("num_sampling_steps must be at least 1")
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)

    def drop_labels(self, labels):
        """返回 CFG dropout 后的类别；labels: [B] -> [B]。"""
        return self.drop_labels_with_mask(labels)[0]

    def drop_labels_with_mask(self, labels):
        """联合 CFG：丢弃类别的样本也关闭 grounding，和 PixelDiT v2 的 null 分支一致。"""
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        return out, ~drop

    def sample_t(self, n, device=None):
        """从 logit-normal 分布采样流匹配时间；输出: [B]。"""
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    @torch.no_grad()
    def encode_grounding(self, noisy_images):
        """只提取当前 z_t 的原始 DINO patch token；projector 在 self.net 中正常反传。"""
        if not self.use_grounding:
            return None
        # 不 clamp noisy_images：z_t 可能超出 [-1,1]，保留其真实噪声分布。
        return self.grounding_encoder(noisy_images)

    def forward(self, x, labels):
        """计算 JiT flow loss；干净图 x 仅构造加噪状态及监督目标，不直接送入 DINO。

        z_t = t*x + (1-t)*noise，DINO 编码 z_t，然后 projector 使用同一个 t。
        c_patch = t_embed + y_embed + projector(DINO(z_t), t)。
        不接受 clean_images/缓存 reference token，避免使用与当前 z_t 不一致的条件。
        """
        if self.training:
            labels_dropped, keep_mask = self.drop_labels_with_mask(labels)
            if self.ground_drop_prob > 0:
                ground_keep = torch.rand(labels.shape[0], device=labels.device) >= self.ground_drop_prob
                keep_mask = keep_mask & ground_keep
        else:
            labels_dropped, keep_mask = labels, None

        # 先构造 noisy image，再提取 grounding；两条分支读取完全相同的 z 和 t。
        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale
        z = t * x + (1 - t) * e
        v = (x - z) / (1 - t).clamp_min(self.t_eps)
        grounding_tokens = self.encode_grounding(z)

        x_pred = self.net(
            z, t.flatten(), labels_dropped,
            grounding_tokens=grounding_tokens,
            grounding_mask=keep_mask if self.use_grounding else None,
        )
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)
        return ((v - v_pred) ** 2).mean(dim=(1, 2, 3)).mean()

    @torch.no_grad()
    def generate(self, labels):
        """仅由类别和随机噪声生成图像；无需原图、验证集或参考 token。

        每次速度评估都从当前 z_t 重算 grounding，包括 Heun 的第二次评估。
        最后一步沿用 JiT 的 Euler 更新，避免在 t=1 处计算速度。
        """
        bsz = labels.size(0)
        z = self.noise_scale * torch.randn(bsz, 3, self.img_size, self.img_size, device=labels.device)
        timesteps = torch.linspace(0.0, 1.0, self.steps + 1, device=labels.device)
        timesteps = timesteps.view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)
        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise ValueError(f"Unsupported sampling method: {self.method}")
        for i in range(self.steps - 1):
            z = stepper(z, timesteps[i], timesteps[i + 1], labels)
        return self._euler_step(z, timesteps[-2], timesteps[-1], labels)

    @torch.no_grad()
    def _forward_sample(self, z, t, labels):
        """用当前 z/t 评估 CFG 速度；不跨状态复用 DINO token。"""
        grounding_tokens = self.encode_grounding(z)
        null_labels = torch.full_like(labels, self.num_classes)
        # Conditional: 当前 noisy DINO token + 当前 t-conditioned projector。
        x_cond = self.net(z, t.flatten(), labels, grounding_tokens=grounding_tokens)
        v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)
        # Unconditional: null label 且关闭 grounding，和训练的联合 CFG dropout 一致。
        x_uncond = self.net(
            z, t.flatten(), null_labels,
            grounding_tokens=grounding_tokens, grounding_mask=False,
        )
        v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)
        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)
        return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels):
        """在 (z_t,t) 评估一次速度，推进到 t_next。"""
        v_pred = self._forward_sample(z, t, labels)
        return z + (t_next - t) * v_pred

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, labels):
        """在 (z_t,t) 和 Euler 预测的 (z_next,t_next) 分别重新编码 grounding。"""
        v_pred_t = self._forward_sample(z, t, labels)
        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, labels)
        return z + (t_next - t) * (0.5 * (v_pred_t + v_pred_t_next))

    @torch.no_grad()
    def update_ema(self):
        """更新 JiT 和 time-conditioned projector 的 EMA；冻结 DINO 不进入 EMA。"""
        source_params = [p for p in self.parameters() if p.requires_grad]
        if self.ema_params1 is None or self.ema_params2 is None:
            raise RuntimeError("EMA buffers must be initialized before update_ema")
        if len(source_params) != len(self.ema_params1) or len(source_params) != len(self.ema_params2):
            raise RuntimeError("EMA buffer does not match trainable parameters")
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
