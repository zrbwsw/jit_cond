"""DinoDiT latent 分支。两阶段复用编码和采样，Stage 1 单独定义训练 loss。"""

import torch
import torch.nn as nn

from .dino_dit import DinoDiT_models
from .dinov2_encoder import DINOLatentEncoder


def sample_logit_normal(batch_size, device, mean, std):
    return torch.sigmoid(torch.randn(batch_size, device=device) * std + mean)


class DinoLatentModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.img_size = args.img_size
        self.num_classes = args.class_num
        self.latent_size = args.img_size // 16
        self.dino_channels = args.grounding_dim
        self.label_drop_prob = args.label_drop_prob
        self.t_mean_dino, self.t_std_dino = args.t_mean_dino, args.t_std_dino
        self.noise_scale_dino = args.noise_scale_dino
        self.t_eps = args.t_eps
        self.steps_dino, self.method_dino = args.steps_dino, args.sampling_method_dino
        self.cfg_dino = args.cfg_dino
        self.cfg_interval_dino = (args.interval_min_dino, args.interval_max_dino)
        self.dino_net = DinoDiT_models[args.dino_model](
            grid_size=self.latent_size, dino_in_channels=self.dino_channels,
            num_classes=args.class_num, attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
        )
        self.dino_trainable_names = tuple(n for n, p in self.dino_net.named_parameters() if p.requires_grad)
        self.dino_encoder = DINOLatentEncoder(
            encoder_type=args.dino_encoder,
            weights_path=args.dinov3_weights or None,
            repo=args.dinov3_repo or None,
            stat_path=args.dino_stat_path or None,
        )
        self.dino_encoder.requires_grad_(False).eval()
        self.ema_decay1, self.ema_decay2 = args.ema_decay1, args.ema_decay2
        self.ema_params1 = self.ema_params2 = None

    def train(self, mode=True):
        super().train(mode)
        self.dino_encoder.eval()
        return self

    def drop_labels_with_mask(self, labels):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        return torch.where(drop, torch.full_like(labels, self.num_classes), labels), ~drop

    @torch.no_grad()
    def encode_grounding(self, clean_images):
        # 先编码 clean RGB，再按统计文件归一化 latent，形状为 [B,768,16,16]。
        latent = self.dino_encoder.encode(clean_images * 0.5 + 0.5)
        expected = (self.dino_channels, self.latent_size, self.latent_size)
        if tuple(latent.shape[1:]) != expected:
            raise ValueError(f"DINO latent shape {tuple(latent.shape[1:])} != {expected}")
        return latent

    @staticmethod
    def _as_tokens(latent):
        return latent.flatten(2).transpose(1, 2).contiguous()

    @torch.no_grad()
    def _velocity_dino(self, z_d, t_d, labels):
        # t_d 是 DinoDiT 自身的流匹配时间，0 为纯噪声，1 为 clean latent。
        x_cond = self.dino_net(z_d, t_d.flatten(), labels)
        denominator = (1 - t_d).clamp_min(self.t_eps)
        v_cond = (x_cond - z_d) / denominator
        if self.cfg_dino == 1.0:
            return v_cond
        null = torch.full_like(labels, self.num_classes)
        x_unc = self.dino_net(z_d, t_d.flatten(), null)
        v_unc = (x_unc - z_d) / denominator
        low, high = self.cfg_interval_dino
        mask = (t_d < high) & ((low == 0) | (t_d > low))
        scale = torch.where(mask, self.cfg_dino, 1.0)
        return v_unc + scale * (v_cond - v_unc)

    @torch.no_grad()
    def sample_dino_latent(self, labels):
        bsz, device = labels.size(0), labels.device
        z_d = self.noise_scale_dino * torch.randn(
            bsz, self.dino_channels, self.latent_size, self.latent_size, device=device)
        ts_d = torch.linspace(0., 1., self.steps_dino + 1, device=device)
        ts_d = ts_d.view(-1, 1, 1, 1, 1).expand(-1, bsz, -1, -1, -1)
        for i in range(self.steps_dino - 1):
            z_d = self._step_dino(z_d, ts_d[i], ts_d[i + 1], labels, self.method_dino == "heun")
        return self._step_dino(z_d, ts_d[-2], ts_d[-1], labels, heun=False)

    def _step_dino(self, z_d, t_d, t_d_next, labels, heun):
        v = self._velocity_dino(z_d, t_d, labels)
        z_euler = z_d + (t_d_next - t_d) * v
        if not heun:
            return z_euler
        v_next = self._velocity_dino(z_euler, t_d_next, labels)
        return z_d + (t_d_next - t_d) * 0.5 * (v + v_next)

    @torch.no_grad()
    def update_ema(self):
        params = [p for p in self.parameters() if p.requires_grad]
        for targets, decay in ((self.ema_params1, self.ema_decay1), (self.ema_params2, self.ema_decay2)):
            if targets is None or len(params) != len(targets):
                raise RuntimeError("EMA buffers do not match trainable parameters")
            for target, param in zip(targets, params):
                target.mul_(decay).add_(param, alpha=1 - decay)


class DinoDenoiser(DinoLatentModel):
    def __init__(self, args):
        super().__init__(args)
        self.loss_scale_dino = args.loss_scale_dino

    def forward(self, x, labels):
        clean = self.encode_grounding(x)
        labels_dropped, _ = self.drop_labels_with_mask(labels)
        # t_d 是 Stage 1 中对归一化 DINO latent 加噪并训练去噪的时间。
        t_d = sample_logit_normal(x.size(0), x.device, self.t_mean_dino, self.t_std_dino).view(-1, 1, 1, 1)
        z_d = t_d * clean + (1 - t_d) * torch.randn_like(clean) * self.noise_scale_dino
        prediction = self.dino_net(z_d, t_d.flatten(), labels_dropped)
        denominator = (1 - t_d).clamp_min(self.t_eps)
        target_v = (clean - z_d) / denominator
        predicted_v = (prediction - z_d) / denominator
        loss_dino = (target_v - predicted_v).square().mean()
        return self.loss_scale_dino * loss_dino, {"loss_dino": loss_dino.item()}
