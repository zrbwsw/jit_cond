"""PixelDiT2 flow matching with clean, noised, or DDT-predicted grounding and REPA."""
import torch
from torch import nn
from tqdm import tqdm

from pixeldit2_c2i import PixelDiT2
from repa import REPATeacher, build_repa_projection, repa_cosine_loss
from stage1.dinov3_k7_encoder import DINOv3Lk7Encoder


def latent_to_tokens(latent):
    """将 RAEv2 的归一化特征图按行展开 [B,1024,16,16] → [B,256,1024]。"""
    if latent.ndim != 4 or tuple(latent.shape[1:]) != (1024, 16, 16):
        raise ValueError(f"Expected normalized RAEv2 latent [B,1024,16,16], got {tuple(latent.shape)}")
    return latent.flatten(2).transpose(1, 2).contiguous()


def _progress_disabled():
    return (torch.distributed.is_available() and torch.distributed.is_initialized()
            and torch.distributed.get_rank() != 0)


class Denoiser(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.img_size != 256:
            raise ValueError("PixelDiT2-H/16 requires 256×256 input images")
        self.img_size = args.img_size
        self.num_classes = args.class_num
        self.condition_mode_pixel = args.condition_mode_pixel
        self.noise_beta_dino_pixel = args.noise_beta_dino_pixel
        self.net = PixelDiT2(num_classes=args.class_num, attn_dropout=args.attn_dropout,
                             proj_dropout=args.proj_dropout,
                             use_t_dino=self.condition_mode_pixel != "clean")
        self.repa_projection = build_repa_projection().requires_grad_(args.repa_weight > 0)
        self.repa_weight = args.repa_weight
        self.repa_align_layer = args.repa_align_layer
        self.dino_encoder = None
        self.repa_encoder = None
        if not args.evaluate_gen:
            self.dino_encoder = DINOv3Lk7Encoder(
                input_size=args.img_size, weights_path=args.dino_encoder_weights_path,
                repo=args.dino_encoder_repo, stat_path=args.dino_stat_path,
            ).requires_grad_(False).eval()
            if self.repa_weight > 0:
                self.repa_encoder = REPATeacher(
                    weights_path=args.repa_encoder_weights_path,
                ).requires_grad_(False).eval()
        self.label_drop_prob = args.label_drop_prob
        self.t_mean_pixel, self.t_std_pixel = args.t_mean_pixel, args.t_std_pixel
        self.noise_scale_pixel, self.t_eps = args.noise_scale_pixel, args.t_eps
        self.steps_pixel, self.cfg_pixel = args.steps_pixel, args.cfg_pixel
        self.cfg_keep_grounding = args.cfg_keep_grounding
        self.cfg_interval_pixel = (args.interval_min_pixel, args.interval_max_pixel)
        self.ema_decay = args.ema_decay
        self.ema_params = None
        self.raev2_checkpoint = args.raev2_checkpoint
        self.raev2_config = args.raev2_config
        self.raev2_root = args.raev2_root
        self.dino_stat_path = args.dino_stat_path
        # Plain Python service, external to the Pixel parameter and state dict trees.
        self.raev2_sampler = None

    def train(self, mode=True):
        super().train(mode)
        for encoder in (self.dino_encoder, self.repa_encoder):
            if encoder is not None:
                encoder.eval()
        return self

    @torch.no_grad()
    def prepare_grounding_pixel(self, clean, labels):
        latent = self.dino_encoder.encode(clean)
        if self.condition_mode_pixel == "clean":
            return latent_to_tokens(latent), None, None
        t_d_p = 1 - self.noise_beta_dino_pixel + self.noise_beta_dino_pixel * torch.rand(
            clean.size(0), device=clean.device)
        t_cond = t_d_p.view(-1, 1, 1, 1)
        latent = t_cond * latent + (1 - t_cond) * torch.randn_like(latent)
        if self.condition_mode_pixel == "pred":
            latent = self._get_raev2_sampler(clean.device).predict_clean(latent, t_d_p, labels)
            t_tag = torch.ones_like(t_d_p)
        else:
            t_tag = t_d_p
        return latent_to_tokens(latent), t_tag, t_d_p

    def forward(self, x, labels):
        if self.dino_encoder is None:
            raise RuntimeError("Training requires the clean DINOv3 grounding encoder")
        clean = (x + 1.0) * 0.5
        grounding, t_dino, t_d_p = self.prepare_grounding_pixel(clean, labels)
        if self.training:
            class_drop = torch.rand(labels.shape, device=labels.device) < self.label_drop_prob
            labels = torch.where(class_drop, self.num_classes, labels)
        t = torch.sigmoid(torch.randn(x.shape[0], device=x.device, dtype=torch.float32)
                          * self.t_std_pixel + self.t_mean_pixel).view(-1, 1, 1, 1)
        noise = torch.randn_like(x) * self.noise_scale_pixel
        x_t = t * x + (1 - t) * noise
        if self.repa_encoder is None:
            prediction = self.net(x_t, t.flatten(), labels, grounding_tokens=grounding, t_dino=t_dino)
        else:
            prediction, features = self.net(x_t, t.flatten(), labels, grounding_tokens=grounding,
                                            repa_align_layer=self.repa_align_layer, t_dino=t_dino)
        scale = (1 - t).clamp_min(self.t_eps)
        velocity = (x - x_t) / scale
        velocity_pred = (prediction - x_t) / scale
        fm_loss = (velocity - velocity_pred).square().mean(dim=(1, 2, 3)).mean()
        metrics = {"fm_loss": fm_loss.item()}
        if t_d_p is not None:
            metrics["t_d_p"] = t_d_p.mean().item()
        if self.repa_encoder is None:
            return fm_loss, metrics
        with torch.no_grad():
            target = self.repa_encoder(clean, self.net.patch_size)
        cos_loss = repa_cosine_loss(self.repa_projection(features), target)
        metrics["cos_loss"] = cos_loss.item()
        return fm_loss + self.repa_weight * cos_loss, metrics

    def release_latent_sampler(self):
        self.raev2_sampler = None

    def _get_raev2_sampler(self, device):
        if self.raev2_sampler is None:
            from raev2_latent_sampler import RAEv2LatentSampler
            self.raev2_sampler = RAEv2LatentSampler(
                self.raev2_checkpoint, self.raev2_config, device,
                root=self.raev2_root, stat_path=self.dino_stat_path)
        return self.raev2_sampler

    @torch.no_grad()
    def generate(self, labels):
        grounding = latent_to_tokens(self._get_raev2_sampler(labels.device).sample_dino_latent(labels))
        grounding_t_dino = (torch.ones(len(labels), device=labels.device)
                            if self.condition_mode_pixel != "clean" else None)
        x = self.noise_scale_pixel * torch.randn(len(labels), 3, self.img_size, self.img_size,
                                                 device=labels.device)
        ts = torch.linspace(0.0, 1.0, self.steps_pixel + 1, device=x.device, dtype=x.dtype)
        ts = ts.view(-1, 1, 1, 1, 1).expand(-1, len(labels), -1, -1, -1)
        null_labels = torch.full_like(labels, self.num_classes)
        with tqdm(total=self.steps_pixel, desc="Pixel DiT sampling", unit="step",
                  dynamic_ncols=True, leave=True, disable=_progress_disabled()) as progress:
            for i in range(self.steps_pixel):
                t, t_next = ts[i], ts[i + 1]
                dt = t_next - t
                v = self._velocity(x, t, labels, null_labels, grounding, grounding_t_dino)
                x_euler = x + dt * v
                if i == self.steps_pixel - 1:
                    progress.update(1)
                    return x_euler
                v_next = self._velocity(x_euler, t_next, labels, null_labels, grounding, grounding_t_dino)
                x = x + dt * 0.5 * (v + v_next)
                progress.update(1)

    def _velocity(self, x, t, labels, null_labels, grounding, t_dino=None):
        scale = (1 - t).clamp_min(self.t_eps)
        condition_kwargs = {"grounding_tokens": grounding}
        if t_dino is not None:
            condition_kwargs["t_dino"] = t_dino
        v_cond = (self.net(x, t.flatten(), labels, **condition_kwargs) - x) / scale
        # 保留 grounding 时，两路共用同一组 patch tokens 和时间条件，只替换类别标签。
        null_kwargs = condition_kwargs if self.cfg_keep_grounding else {"skip_grounding": True}
        v_null = (self.net(x, t.flatten(), null_labels, **null_kwargs) - x) / scale
        low, high = self.cfg_interval_pixel
        inside = (t < high) & ((t > low) | (low <= 0))
        guidance = torch.where(inside, self.cfg_pixel, 1.0)
        return v_null + guidance * (v_cond - v_null)

    @torch.no_grad()
    def update_ema(self):
        if self.ema_params is None:
            raise RuntimeError("EMA has not been initialized")
        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                self.ema_params[name].lerp_(parameter.detach(), 1.0 - self.ema_decay)
