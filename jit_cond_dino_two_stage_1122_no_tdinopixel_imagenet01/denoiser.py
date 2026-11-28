"""Stage 2 Pixel JiT，以冻结的 Stage 1 latent 分支提供 grounding。"""

import torch

from model_jit import JiT_models
from stage1.denoiser_dino import DinoLatentModel, sample_logit_normal


class Denoiser(DinoLatentModel):
    def __init__(self, args):
        super().__init__(args)
        self.dino_net.requires_grad_(False).eval()
        self.net = JiT_models[args.model](
            input_size=args.img_size, in_channels=3, num_classes=args.class_num,
            attn_drop=args.attn_dropout, proj_drop=args.proj_dropout,
            grounding_dim=args.grounding_dim,
        )
        self.condition_mode_pixel = args.condition_mode_pixel
        self.noise_beta_dino_pixel = args.noise_beta_dino_pixel
        self.t_schedule_dino_pixel = args.t_schedule_dino_pixel
        self.t_mean_pixel, self.t_std_pixel = args.t_mean_pixel, args.t_std_pixel
        self.noise_scale_pixel = args.noise_scale_pixel
        self.steps_pixel, self.method_pixel = args.steps_pixel, args.sampling_method_pixel
        self.cfg_pixel = args.cfg_pixel
        self.cfg_interval_pixel = (args.interval_min_pixel, args.interval_max_pixel)

    def train(self, mode=True):
        super().train(mode)
        self.dino_net.eval()
        return self

    @torch.no_grad()
    def prepare_grounding_pixel(self, x, labels):
        clean = self.encode_grounding(x)
        if self.condition_mode_pixel == "clean":
            return self._as_tokens(clean), None
        # t_d_p 控制 Stage 2 grounding 的加噪程度，与像素去噪时间 t_p 独立。
        if self.t_schedule_dino_pixel == "uniform":
            beta = self.noise_beta_dino_pixel
            t_d_p = 1 - beta + beta * torch.rand(x.size(0), device=x.device)
        else:
            t_d_p = sample_logit_normal(x.size(0), x.device, self.t_mean_dino, self.t_std_dino)
        t_cond = t_d_p.view(-1, 1, 1, 1)
        z_cond = t_cond * clean + (1 - t_cond) * torch.randn_like(clean) * self.noise_scale_dino
        if self.condition_mode_pixel == "pred":
            # 冻结的 DinoDiT 用实际加噪时间做一次 clean 端点预测。
            z_cond = self.dino_net(z_cond, t_d_p, labels)
        return self._as_tokens(z_cond), t_d_p

    def forward(self, x, labels):
        grounding_tokens, t_d_p = self.prepare_grounding_pixel(x, labels)
        labels_dropped, keep_mask = self.drop_labels_with_mask(labels)
        # t_p 是像素流匹配时间，仅用它对 RGB 图像加噪。
        t_p = sample_logit_normal(x.size(0), x.device, self.t_mean_pixel, self.t_std_pixel).view(-1, 1, 1, 1)
        z_p = t_p * x + (1 - t_p) * torch.randn_like(x) * self.noise_scale_pixel
        x_pred = self.net(
            z_p, t_p.flatten(), labels_dropped, grounding_tokens=grounding_tokens,
            grounding_mask=keep_mask,
        )
        denominator = (1 - t_p).clamp_min(self.t_eps)
        target_v = (x - z_p) / denominator
        predicted_v = (x_pred - z_p) / denominator
        loss_pixel = (target_v - predicted_v).square().mean()
        metrics = {"loss_pixel": loss_pixel.item()}
        if t_d_p is not None:
            metrics["t_d_p"] = t_d_p.mean().item()
        return loss_pixel, metrics

    @torch.no_grad()
    def generate(self, labels, grounding_tokens=None):
        if grounding_tokens is None:
            grounding_tokens = self._as_tokens(self.sample_dino_latent(labels))
        bsz = labels.size(0)
        z_p = self.noise_scale_pixel * torch.randn(bsz, 3, self.img_size, self.img_size, device=labels.device)
        ts_p = torch.linspace(0., 1., self.steps_pixel + 1, device=labels.device)
        ts_p = ts_p.view(-1, 1, 1, 1, 1).expand(-1, bsz, -1, -1, -1)
        for i in range(self.steps_pixel - 1):
            z_p = self._step_pixel(z_p, ts_p[i], ts_p[i + 1], labels, grounding_tokens,
                                   self.method_pixel == "heun")
        return self._step_pixel(z_p, ts_p[-2], ts_p[-1], labels, grounding_tokens,
                                heun=False)

    def _velocity_pixel(self, z_p, t_p, labels, grounding_tokens):
        null_labels = torch.full_like(labels, self.num_classes)
        x_cond = self.net(z_p, t_p.flatten(), labels, grounding_tokens=grounding_tokens)
        x_uncond = self.net(z_p, t_p.flatten(), null_labels, grounding_tokens=grounding_tokens,
                            grounding_mask=False)
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
