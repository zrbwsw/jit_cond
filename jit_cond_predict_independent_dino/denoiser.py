"""Pixel JiT trained on single predictions from a frozen pretrained DinoDiT."""

import torch

from denoiser_dino import DinoDenoiser
from model_jit import JiT_models


class Denoiser(DinoDenoiser):
    def __init__(self, args):
        super().__init__(args)
        self.dino_net.requires_grad_(False).eval()
        self.net = JiT_models[args.model](
            input_size=args.img_size, in_channels=3, num_classes=args.class_num,
            attn_drop=args.attn_dropout, proj_drop=args.proj_dropout,
            grounding_dim=args.grounding_dim, use_t_dino=True,
        )
        self.use_grounding = True
        self.dino_noise_beta = args.dino_noise_beta
        self.condition_time_schedule = args.condition_time_schedule
        self.method, self.steps = args.sampling_method, args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)

    def train(self, mode=True):
        super().train(mode)
        self.dino_net.eval()
        return self

    @torch.no_grad()
    def predict_grounding(self, x, labels):
        clean = self.encode_grounding(x)
        if self.condition_time_schedule == "uniform":
            s = 1 - self.dino_noise_beta + self.dino_noise_beta * torch.rand(x.size(0), device=x.device)
        else:
            s = self.sample_t(x.size(0), x.device, self.D_mean, self.D_std)
        s_image = s.view(-1, 1, 1, 1)
        noisy = s_image * clean + (1 - s_image) * torch.randn_like(clean) * self.noise_scale
        # Exactly one conditional forward, not a velocity or a multi-step rollout.
        predicted_clean = self.dino_net(noisy, s, labels)
        return self._as_tokens(predicted_clean), s

    def forward(self, x, labels):
        grounding_tokens, condition_time = self.predict_grounding(x, labels)
        labels_dropped, keep_mask = self.drop_labels_with_mask(labels)
        t = self.sample_t(x.size(0), device=x.device).view(-1, 1, 1, 1)
        z = t * x + (1 - t) * torch.randn_like(x) * self.noise_scale
        # s describes DinoDiT's INPUT. Pixel receives its CLEAN prediction, hence tag=1.
        x_pred = self.net(
            z, t.flatten(), labels_dropped, grounding_tokens=grounding_tokens,
            grounding_mask=keep_mask, t_dino=torch.ones_like(t.flatten()),
        )
        denominator = (1 - t).clamp_min(self.t_eps)
        target_v = (x - z) / denominator
        predicted_v = (x_pred - z) / denominator
        loss = (target_v - predicted_v).square().mean()
        return loss, {"loss_jit": loss.item(), "condition_time": condition_time.mean().item()}

    @torch.no_grad()
    def generate(self, labels, grounding_tokens=None):
        if grounding_tokens is None:
            grounding_tokens = self._as_tokens(self.sample_dino_latent(labels))
        bsz = labels.size(0)
        z = self.noise_scale * torch.randn(bsz, 3, self.img_size, self.img_size, device=labels.device)
        ts = torch.linspace(0., 1., self.steps + 1, device=labels.device)
        ts = ts.view(-1, 1, 1, 1, 1).expand(-1, bsz, -1, -1, -1)
        stepper = self._heun_step if self.method == "heun" else self._euler_step
        for i in range(self.steps - 1):
            z = stepper(z, ts[i], ts[i + 1], labels, grounding_tokens)
        return self._euler_step(z, ts[-2], ts[-1], labels, grounding_tokens)

    @torch.no_grad()
    def _forward_sample(self, z, t, labels, grounding_tokens=None):
        null_labels = torch.full_like(labels, self.num_classes)
        t_dino = torch.ones_like(t.flatten())
        x_cond = self.net(z, t.flatten(), labels, grounding_tokens=grounding_tokens, t_dino=t_dino)
        v_cond = (x_cond - z) / (1 - t).clamp_min(self.t_eps)
        x_uncond = self.net(z, t.flatten(), null_labels, grounding_tokens=grounding_tokens,
                            grounding_mask=False, t_dino=t_dino)
        v_uncond = (x_uncond - z) / (1 - t).clamp_min(self.t_eps)
        low, high = self.cfg_interval
        mask = (t < high) & ((low == 0) | (t > low))
        scale = torch.where(mask, self.cfg_scale, 1.0)
        return v_uncond + scale * (v_cond - v_uncond)

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels, grounding_tokens=None):
        return z + (t_next - t) * self._forward_sample(z, t, labels, grounding_tokens)

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, labels, grounding_tokens=None):
        v = self._forward_sample(z, t, labels, grounding_tokens)
        z_euler = z + (t_next - t) * v
        v_next = self._forward_sample(z_euler, t_next, labels, grounding_tokens)
        return z + (t_next - t) * 0.5 * (v + v_next)
