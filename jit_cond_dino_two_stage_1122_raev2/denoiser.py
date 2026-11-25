"""Stage 2 Pixel JiT，以冻结的 DINOv3-L k=7 encoder 提供 grounding。

与原版的差异：
- 使用 RAEv2 的 DINOv3-L k=7 encoder（无需训练 Stage 1）
- Latent 维度从 768 升级到 1024
- 训练时条件：clean image → dinov3_k7 → normalized latent
- 推理时条件：需要调用 RAEv2 Stage 1 采样 latent
"""

import torch

from model_jit import JiT_models
from stage1.dinov3_k7_encoder import DINOv3Lk7Encoder


def sample_logit_normal(batch_size, device, mean, std):
    return torch.sigmoid(torch.randn(batch_size, device=device) * std + mean)


class Denoiser(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        # DINOv3-L k=7 encoder（冻结）
        self.dino_encoder = DINOv3Lk7Encoder(
            input_size=args.img_size,
            weights_path=getattr(args, 'dino_encoder_weights_path', None),
            repo=getattr(args, 'dino_encoder_repo', None),
            stat_path=args.dino_stat_path,
        )
        self.dino_encoder.requires_grad_(False).eval()

        self.grounding_dim = self.dino_encoder.embed_dim  # 1024
        self.latent_size = args.img_size // 16

        # Pixel JiT
        self.net = JiT_models[args.model](
            input_size=args.img_size, in_channels=3, num_classes=args.class_num,
            attn_drop=args.attn_dropout, proj_drop=args.proj_dropout,
            grounding_dim=self.grounding_dim, use_t_dino=False,  # RAEv2 不需要 t_dino
        )

        # Pixel JiT 参数
        self.img_size = args.img_size
        self.num_classes = args.class_num
        self.label_drop_prob = args.label_drop_prob
        self.t_mean_pixel, self.t_std_pixel = args.t_mean_pixel, args.t_std_pixel
        self.noise_scale_pixel = args.noise_scale_pixel
        self.t_eps = args.t_eps
        self.steps_pixel, self.method_pixel = args.steps_pixel, args.sampling_method_pixel
        self.cfg_pixel = args.cfg_pixel
        self.cfg_interval_pixel = (args.interval_min_pixel, args.interval_max_pixel)

        # EMA
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

    def train(self, mode=True):
        super().train(mode)
        self.dino_encoder.eval()  # 始终冻结
        return self

    def drop_labels_with_mask(self, labels):
        """Label dropout for CFG."""
        if self.training and self.label_drop_prob > 0:
            mask = torch.rand(labels.size(0), device=labels.device) >= self.label_drop_prob
            return torch.where(mask, labels, self.num_classes), mask
        return labels, torch.ones(labels.size(0), dtype=torch.bool, device=labels.device)

    @torch.no_grad()
    def encode_grounding(self, x):
        """提取 clean image 的 DINOv3-L k=7 latent。"""
        return self.dino_encoder.encode(x)

    @torch.no_grad()
    def _as_tokens(self, z):
        """将 latent [B,C,H,W] 转为 token 序列 [B,N,C]。"""
        return z.flatten(2).transpose(1, 2).contiguous()

    @torch.no_grad()
    def prepare_grounding_pixel(self, x, labels):
        """准备 Pixel JiT 的 grounding 条件。

        训练时：直接使用 clean latent（最强实例信息）
        推理时：需要外部提供 RAEv2 采样的 latent
        """
        clean_latent = self.encode_grounding(x)  # [B, 1024, 16, 16]
        grounding_tokens = self._as_tokens(clean_latent)  # [B, 256, 1024]
        return grounding_tokens

    def _velocity_pixel(self, z_p, t_p, labels, grounding_tokens):
        """Pixel JiT 的速度场预测。"""
        labels_dropped, _ = self.drop_labels_with_mask(labels)
        x_pred = self.net(z_p, t_p.flatten(), labels_dropped, grounding_tokens)
        denominator = (1 - t_p).clamp_min(self.t_eps)
        return (x_pred - z_p) / denominator

    def _step_pixel(self, z_p, t_p, t_p_next, labels, grounding_tokens, heun):
        """Pixel JiT ODE 积分一步（Euler 或 Heun）。"""
        v = self._velocity_pixel(z_p, t_p, labels, grounding_tokens)
        z_euler = z_p + (t_p_next - t_p) * v
        if not heun:
            return z_euler
        v_next = self._velocity_pixel(z_euler, t_p_next, labels, grounding_tokens)
        return z_p + (t_p_next - t_p) * 0.5 * (v + v_next)

    @torch.no_grad()
    def sample_pixel(self, labels, grounding_tokens):
        """Pixel JiT 采样（从噪声到图像）。

        Args:
            labels: [B] class labels
            grounding_tokens: [B, N, C] 预先计算的 latent tokens
                - 推理时：从 RAEv2 Stage 1 采样得到
                - 测试时：可以用 clean image 的 latent
        """
        bsz, device = len(labels), labels.device
        z_p = torch.randn(bsz, 3, self.img_size, self.img_size, device=device) * self.noise_scale_pixel

        ts_p = torch.linspace(0., 1., self.steps_pixel + 1, device=device)
        ts_p = ts_p.view(-1, 1, 1, 1, 1).expand(-1, bsz, -1, -1, -1)

        for i in range(self.steps_pixel - 1):
            z_p = self._step_pixel(z_p, ts_p[i], ts_p[i + 1], labels, grounding_tokens, self.method_pixel == "heun")
        return self._step_pixel(z_p, ts_p[-2], ts_p[-1], labels, grounding_tokens, heun=False)

    def forward(self, x, labels):
        """Stage 2 训练：Pixel JiT v-loss。"""
        # 准备 grounding（clean latent）
        grounding_tokens = self.prepare_grounding_pixel(x, labels)

        # 像素加噪
        t_p = sample_logit_normal(x.size(0), x.device, self.t_mean_pixel, self.t_std_pixel).view(-1, 1, 1, 1)
        z_p = t_p * x + (1 - t_p) * torch.randn_like(x) * self.noise_scale_pixel

        # 预测
        labels_dropped, _ = self.drop_labels_with_mask(labels)
        x_pred = self.net(z_p, t_p.flatten(), labels_dropped, grounding_tokens)

        # v-loss
        denominator = (1 - t_p).clamp_min(self.t_eps)
        target_v = (x - z_p) / denominator
        predicted_v = (x_pred - z_p) / denominator
        loss_pixel = (target_v - predicted_v).square().mean()

        return loss_pixel, {"loss_pixel": loss_pixel.item()}

    @torch.no_grad()
    def update_ema(self):
        """更新 EMA 参数（仅 Pixel JiT 网络）。"""
        if self.ema_params1 is None:
            self.ema_params1 = [p.clone().detach() for p in self.net.parameters()]
            self.ema_params2 = [p.clone().detach() for p in self.net.parameters()]

        for targ, src in zip(self.ema_params1, self.net.parameters()):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, self.net.parameters()):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)


    @torch.no_grad()
    def prepare_grounding_pixel(self, x, labels):
        clean = self.encode_grounding(x)
        # t_d_p 控制 Stage 2 grounding 的加噪程度，与像素去噪时间 t_p 独立。
        if self.t_schedule_dino_pixel == "uniform":
            beta = self.noise_beta_dino_pixel
            t_d_p = 1 - beta + beta * torch.rand(x.size(0), device=x.device)
        else:
            t_d_p = sample_logit_normal(x.size(0), x.device, self.t_mean_dino, self.t_std_dino)
        t_cond = t_d_p.view(-1, 1, 1, 1)
        z_cond = t_cond * clean + (1 - t_cond) * torch.randn_like(clean) * self.noise_scale_dino
        if self.condition_mode_pixel == "pred":
            # 一次 x-prediction 返回 clean 端点估计，Pixel 的时间标记取 1。
            z_cond = self.dino_net(z_cond, t_d_p, labels)
            t_tag = torch.ones_like(t_d_p)
        else:
            # 直接注入带噪 latent，让 Pixel 看到其真实的加噪时间。
            t_tag = t_d_p
        return self._as_tokens(z_cond), t_tag, t_d_p

    def forward(self, x, labels):
        grounding_tokens, t_tag, t_d_p = self.prepare_grounding_pixel(x, labels)
        labels_dropped, keep_mask = self.drop_labels_with_mask(labels)
        # t_p 是像素流匹配时间，仅用它对 RGB 图像加噪。
        t_p = sample_logit_normal(x.size(0), x.device, self.t_mean_pixel, self.t_std_pixel).view(-1, 1, 1, 1)
        z_p = t_p * x + (1 - t_p) * torch.randn_like(x) * self.noise_scale_pixel
        x_pred = self.net(
            z_p, t_p.flatten(), labels_dropped, grounding_tokens=grounding_tokens,
            grounding_mask=keep_mask, t_dino=t_tag,
        )
        denominator = (1 - t_p).clamp_min(self.t_eps)
        target_v = (x - z_p) / denominator
        predicted_v = (x_pred - z_p) / denominator
        loss_pixel = (target_v - predicted_v).square().mean()
        return loss_pixel, {"loss_pixel": loss_pixel.item(), "t_d_p": t_d_p.mean().item()}

    @torch.no_grad()
    def generate(self, labels, grounding_tokens=None):
        if grounding_tokens is None:
            grounding_tokens = self._as_tokens(self.sample_dino_latent(labels))
        bsz = labels.size(0)
        z_p = self.noise_scale_pixel * torch.randn(bsz, 3, self.img_size, self.img_size, device=labels.device)
        ts_p = torch.linspace(0., 1., self.steps_pixel + 1, device=labels.device)
        ts_p = ts_p.view(-1, 1, 1, 1, 1).expand(-1, bsz, -1, -1, -1)
        for i in range(self.steps_pixel - 1):
            z_p = self._step_pixel(z_p, ts_p[i], ts_p[i + 1], labels, grounding_tokens, self.method_pixel == "heun")
        return self._step_pixel(z_p, ts_p[-2], ts_p[-1], labels, grounding_tokens, heun=False)

    def _velocity_pixel(self, z_p, t_p, labels, grounding_tokens):
        null_labels = torch.full_like(labels, self.num_classes)
        # 两种训练模式采样时都使用已积分到 clean 端的 DINO latent。
        t_tag = torch.ones_like(t_p.flatten())
        x_cond = self.net(z_p, t_p.flatten(), labels, grounding_tokens=grounding_tokens, t_dino=t_tag)
        x_uncond = self.net(z_p, t_p.flatten(), null_labels, grounding_tokens=grounding_tokens,
                            grounding_mask=False, t_dino=t_tag)
        denominator = (1 - t_p).clamp_min(self.t_eps)
        v_cond = (x_cond - z_p) / denominator
        v_uncond = (x_uncond - z_p) / denominator
        low, high = self.cfg_interval_pixel
        mask = (t_p < high) & ((low == 0) | (t_p > low))
        scale = torch.where(mask, self.cfg_pixel, 1.0)
        return v_uncond + scale * (v_cond - v_uncond)

    def _step_pixel(self, z_p, t_p, t_p_next, labels, grounding_tokens, heun):
        v = self._velocity_pixel(z_p, t_p, labels, grounding_tokens)
        z_euler = z_p + (t_p_next - t_p) * v
        if not heun:
            return z_euler
        v_next = self._velocity_pixel(z_euler, t_p_next, labels, grounding_tokens)
        return z_p + (t_p_next - t_p) * 0.5 * (v + v_next)
