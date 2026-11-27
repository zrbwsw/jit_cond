"""Pixel JiT with clean, noised, or DDT-predicted RAEv2 grounding."""
import torch

from model_jit import JiT_models
from stage1.dinov3_k7_encoder import DINOv3Lk7Encoder


def sample_logit_normal(batch_size, device, mean, std):
    return torch.sigmoid(torch.randn(batch_size, device=device) * std + mean)


class Denoiser(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dino_encoder = DINOv3Lk7Encoder(
            input_size=args.img_size,
            weights_path=getattr(args, "dino_encoder_weights_path", None),
            repo=getattr(args, "dino_encoder_repo", None),
            stat_path=args.dino_stat_path,
        )
        self.dino_encoder.requires_grad_(False).eval()
        self.grounding_dim = self.dino_encoder.embed_dim
        self.latent_size = args.img_size // 16

        self.condition_mode_pixel = args.condition_mode_pixel
        self.noise_beta_dino_pixel = args.noise_beta_dino_pixel
        self.net = JiT_models[args.model](
            input_size=args.img_size, in_channels=3, num_classes=args.class_num,
            attn_drop=args.attn_dropout, proj_drop=args.proj_dropout,
            grounding_dim=self.grounding_dim,
            use_t_dino=self.condition_mode_pixel != "clean",
        )
        self.img_size = args.img_size
        self.num_classes = args.class_num
        self.label_drop_prob = args.label_drop_prob
        self.t_mean_pixel, self.t_std_pixel = args.t_mean_pixel, args.t_std_pixel
        self.noise_scale_pixel = args.noise_scale_pixel
        self.t_eps = args.t_eps
        self.steps_pixel, self.method_pixel = args.steps_pixel, args.sampling_method_pixel
        self.cfg_pixel = args.cfg_pixel
        self.cfg_interval_pixel = (args.interval_min_pixel, args.interval_max_pixel)
        self.method_dino = "raev2"
        self.steps_dino = 100
        self.cfg_dino = 1.0
        self.cfg_interval_dino = (0.0, 1.0)
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None
        self.raev2_checkpoint = getattr(args, "raev2_checkpoint", "")
        self.raev2_config = getattr(args, "raev2_config", "")
        self.raev2_sampler = None

    def train(self, mode=True):
        super().train(mode)
        self.dino_encoder.eval()
        return self

    def drop_labels_with_mask(self, labels):
        if self.training and self.label_drop_prob > 0:
            mask = torch.rand(labels.size(0), device=labels.device) >= self.label_drop_prob
            return torch.where(mask, labels, self.num_classes), mask
        return labels, torch.ones(labels.size(0), dtype=torch.bool, device=labels.device)

    @torch.no_grad()
    def encode_grounding(self, x):
        return self.dino_encoder.encode(x)

    @torch.no_grad()
    def _as_tokens(self, z):
        return z.flatten(2).transpose(1, 2).contiguous()

    @torch.no_grad()
    def prepare_grounding_pixel(self, x, labels):
        clean = self.encode_grounding(x)
        if self.condition_mode_pixel == "clean":
            return self._as_tokens(clean), None, None
        t_d_p = 1 - self.noise_beta_dino_pixel + self.noise_beta_dino_pixel * torch.rand(
            x.size(0), device=x.device)
        t_cond = t_d_p.view(-1, 1, 1, 1)
        z_cond = t_cond * clean + (1 - t_cond) * torch.randn_like(clean)
        if self.condition_mode_pixel == "pred":
            z_cond = self._get_raev2_sampler(x.device).predict_clean(z_cond, t_d_p, labels)
            t_tag = torch.ones_like(t_d_p)
        else:
            t_tag = t_d_p
        return self._as_tokens(z_cond), t_tag, t_d_p

    def _get_raev2_sampler(self, device):
        if not self.raev2_checkpoint or not self.raev2_config:
            raise ValueError("RAEv2 sampling requires --raev2_checkpoint and --raev2_config")
        if self.raev2_sampler is None:
            from raev2_latent_sampler import RAEv2LatentSampler
            self.raev2_sampler = RAEv2LatentSampler(
                self.raev2_checkpoint, self.raev2_config, device)
        return self.raev2_sampler

    @torch.no_grad()
    def sample_dino_latent(self, labels):
        return self._get_raev2_sampler(labels.device).sample_dino_latent(labels)

    def forward(self, x, labels):
        grounding_tokens, t_dino, t_d_p = self.prepare_grounding_pixel(x, labels)
        labels_dropped, keep_mask = self.drop_labels_with_mask(labels)
        t_p = sample_logit_normal(x.size(0), x.device, self.t_mean_pixel, self.t_std_pixel).view(-1, 1, 1, 1)
        z_p = t_p * x + (1 - t_p) * torch.randn_like(x) * self.noise_scale_pixel
        x_pred = self.net(
            z_p, t_p.flatten(), labels_dropped,
            grounding_tokens=grounding_tokens, grounding_mask=keep_mask, t_dino=t_dino,
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
        grounding_t_dino = (torch.ones(bsz, device=labels.device)
                            if self.condition_mode_pixel != "clean" else None)
        z_p = self.noise_scale_pixel * torch.randn(bsz, 3, self.img_size, self.img_size, device=labels.device)
        ts_p = torch.linspace(0., 1., self.steps_pixel + 1, device=labels.device)
        ts_p = ts_p.view(-1, 1, 1, 1, 1).expand(-1, bsz, -1, -1, -1)
        for i in range(self.steps_pixel - 1):
            z_p = self._step_pixel(z_p, ts_p[i], ts_p[i + 1], labels, grounding_tokens,
                                   self.method_pixel == "heun", grounding_t_dino)
        return self._step_pixel(z_p, ts_p[-2], ts_p[-1], labels, grounding_tokens,
                                heun=False, grounding_t_dino=grounding_t_dino)

    def _velocity_pixel(self, z_p, t_p, labels, grounding_tokens, grounding_t_dino=None):
        null_labels = torch.full_like(labels, self.num_classes)
        x_cond = self.net(
            z_p, t_p.flatten(), labels,
            grounding_tokens=grounding_tokens, t_dino=grounding_t_dino,
        )
        x_uncond = self.net(
            z_p, t_p.flatten(), null_labels,
            grounding_tokens=grounding_tokens, grounding_mask=False, t_dino=grounding_t_dino,
        )
        denominator = (1 - t_p).clamp_min(self.t_eps)
        v_cond = (x_cond - z_p) / denominator
        v_uncond = (x_uncond - z_p) / denominator
        low, high = self.cfg_interval_pixel
        mask = (t_p < high) & ((low == 0) | (t_p > low))
        scale = torch.where(mask, self.cfg_pixel, 1.0)
        return v_uncond + scale * (v_cond - v_uncond)

    def _step_pixel(self, z_p, t_p, t_p_next, labels, grounding_tokens, heun, grounding_t_dino=None):
        v = self._velocity_pixel(z_p, t_p, labels, grounding_tokens, grounding_t_dino)
        z_euler = z_p + (t_p_next - t_p) * v
        if not heun:
            return z_euler
        v_next = self._velocity_pixel(z_euler, t_p_next, labels, grounding_tokens, grounding_t_dino)
        return z_p + (t_p_next - t_p) * 0.5 * (v + v_next)

    @torch.no_grad()
    def update_ema(self):
        if self.ema_params1 is None or self.ema_params2 is None:
            raise RuntimeError("EMA parameters are not initialized")
        params = [parameter for parameter in self.parameters() if parameter.requires_grad]
        if len(params) != len(self.ema_params1) or len(params) != len(self.ema_params2):
            raise ValueError("EMA parameters do not match trainable parameters")
        for target, source in zip(self.ema_params1, params):
            target.detach().mul_(self.ema_decay1).add_(source, alpha=1 - self.ema_decay1)
        for target, source in zip(self.ema_params2, params):
            target.detach().mul_(self.ema_decay2).add_(source, alpha=1 - self.ema_decay2)
