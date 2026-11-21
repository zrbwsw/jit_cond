"""Standalone DINO latent flow training and the shared DINO sampler."""

import torch
import torch.nn as nn

from latent_force.dino_dit import DinoDiT_models
from latent_force.dinov2_encoder import DINOLatentEncoder


class DinoDenoiser(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.img_size = args.img_size
        self.num_classes = args.class_num
        self.latent_size = args.img_size // 16
        self.dino_channels = args.grounding_dim
        self.label_drop_prob = args.label_drop_prob
        self.P_mean, self.P_std = args.P_mean, args.P_std
        self.D_mean, self.D_std = args.D_mean, args.D_std
        self.noise_scale, self.t_eps = args.noise_scale, args.t_eps
        self.dino_loss_scale = args.dino_loss_scale
        self.dino_steps = args.dino_steps
        self.dino_method = args.dino_sampling_method
        self.dino_cfg_scale = args.dino_cfg
        self.dino_cfg_interval = (args.dino_interval_min, args.dino_interval_max)
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

    def sample_t(self, n, device=None, mean=None, std=None):
        mean = self.P_mean if mean is None else mean
        std = self.P_std if std is None else std
        return torch.sigmoid(torch.randn(n, device=device) * std + mean)

    @torch.no_grad()
    def encode_grounding(self, clean_images):
        latent = self.dino_encoder.encode(clean_images * 0.5 + 0.5)
        expected = (self.dino_channels, self.latent_size, self.latent_size)
        if tuple(latent.shape[1:]) != expected:
            raise ValueError(f"DINO latent shape {tuple(latent.shape[1:])} != {expected}")
        return latent

    @staticmethod
    def _as_tokens(latent):
        return latent.flatten(2).transpose(1, 2).contiguous()

    def forward(self, x, labels):
        clean = self.encode_grounding(x)
        labels_dropped, _ = self.drop_labels_with_mask(labels)
        t = self.sample_t(x.size(0), x.device, self.D_mean, self.D_std).view(-1, 1, 1, 1)
        noisy = t * clean + (1 - t) * torch.randn_like(clean) * self.noise_scale
        prediction = self.dino_net(noisy, t.flatten(), labels_dropped)
        # Same velocity objective as join-dino; no Pixel model is constructed.
        denominator = (1 - t).clamp_min(self.t_eps)
        target_v = (clean - noisy) / denominator
        predicted_v = (prediction - noisy) / denominator
        loss = (target_v - predicted_v).square().mean()
        return self.dino_loss_scale * loss, {"loss_dino": loss.item()}

    @torch.no_grad()
    def _dino_velocity(self, z_d, t_d, labels):
        x_cond = self.dino_net(z_d, t_d.flatten(), labels)
        v_cond = (x_cond - z_d) / (1 - t_d).clamp_min(self.t_eps)
        if self.dino_cfg_scale == 1.0:
            return v_cond
        null = torch.full_like(labels, self.num_classes)
        x_unc = self.dino_net(z_d, t_d.flatten(), null)
        v_unc = (x_unc - z_d) / (1 - t_d).clamp_min(self.t_eps)
        low, high = self.dino_cfg_interval
        mask = (t_d < high) & ((low == 0) | (t_d > low))
        scale = torch.where(mask, self.dino_cfg_scale, 1.0)
        return v_unc + scale * (v_cond - v_unc)

    @torch.no_grad()
    def sample_dino_latent(self, labels):
        bsz, device = labels.size(0), labels.device
        z = self.noise_scale * torch.randn(
            bsz, self.dino_channels, self.latent_size, self.latent_size, device=device)
        ts = torch.linspace(0., 1., self.dino_steps + 1, device=device)
        ts = ts.view(-1, 1, 1, 1, 1).expand(-1, bsz, -1, -1, -1)
        for i in range(self.dino_steps - 1):
            z = self._dino_step(z, ts[i], ts[i + 1], labels, self.dino_method == "heun")
        return self._dino_step(z, ts[-2], ts[-1], labels, heun=False)

    @torch.no_grad()
    def _dino_step(self, z, t, t_next, labels, heun):
        v = self._dino_velocity(z, t, labels)
        z_euler = z + (t_next - t) * v
        if not heun:
            return z_euler
        v2 = self._dino_velocity(z_euler, t_next, labels)
        return z + (t_next - t) * 0.5 * (v + v2)

    @torch.no_grad()
    def update_ema(self):
        params = [p for p in self.parameters() if p.requires_grad]
        for targets, decay in ((self.ema_params1, self.ema_decay1),
                               (self.ema_params2, self.ema_decay2)):
            if targets is None or len(params) != len(targets):
                raise RuntimeError("EMA buffers do not match trainable parameters")
            for target, param in zip(targets, params):
                target.mul_(decay).add_(param, alpha=1 - decay)
