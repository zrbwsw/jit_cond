"""Sample normalized RAEv2 latents with the official stage-2 EMA model."""
import math
from pathlib import Path

import torch
from omegaconf import OmegaConf

from configs.stage2 import Stage2Config
from stage2.transport import create_sampler, create_transport
from utils.guidance_utils import get_model_forward_fn
from utils.model_utils import instantiate_from_config


class RAEv2LatentSampler:
    def __init__(self, checkpoint_path, config_path, device):
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        config_path = Path(config_path).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"RAEv2 checkpoint not found: {checkpoint_path}")
        if not config_path.is_file():
            raise FileNotFoundError(f"RAEv2 sampling config not found: {config_path}")

        config = OmegaConf.to_object(OmegaConf.merge(
            OmegaConf.structured(Stage2Config), OmegaConf.load(config_path)))
        config.post_process()
        config.stage_2.ckpt = str(checkpoint_path)
        config.prepare_model_params()
        self.config = config
        self.device = torch.device(device)
        self.latent_size = tuple(config.misc.latent_size)
        self.null_label = config.misc.num_classes
        self.model = instantiate_from_config(config.stage_2).to(self.device).eval()
        self.model_fn, self.sample_model_kwargs = get_model_forward_fn(self.model, config.guidance)
        self.use_guidance = config.guidance.any_guidance_active

        time_dist_shift = math.sqrt(
            math.prod(self.latent_size) / config.misc.time_dist_shift_base)
        transport = create_transport(config=config.transport, time_dist_shift=time_dist_shift)
        transport_sampler = create_sampler(transport, guidance_config=config.guidance)
        self.sample_fn = transport_sampler.sample_ode(num_steps=config.sampler.num_steps)
        print(
            f"RAEv2 sampler ready: steps={config.sampler.num_steps}, "
            f"CFG={config.guidance.cfg.scale}, IG={config.guidance.ig.scale}, "
            f"latent_size={self.latent_size}", flush=True)

    @torch.no_grad()
    def predict_clean(self, z_cond, t_dino_pixel, labels):
        expected = (len(labels), *self.latent_size)
        if z_cond.shape != expected or t_dino_pixel.shape != (len(labels),):
            raise ValueError("Unexpected noised latent or t_dino_pixel shape")
        z_cond = z_cond.to(self.device)
        labels = labels.to(self.device)
        t_dino_pixel = t_dino_pixel.to(self.device)
        # RAEv2 uses xt=(1-t)*clean+t*noise, while t_dino_pixel is the clean weight.
        # z_cond is already constructed, so do not apply another time-dist shift.
        rae_t = 1 - t_dino_pixel
        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            prediction = self.model(z_cond, rae_t, context=labels, attn_mask=None)
        if isinstance(prediction, tuple):
            prediction = prediction[0]
        if prediction.shape != expected:
            raise ValueError(f"Unexpected RAEv2 prediction shape: {prediction.shape}")
        if not prediction.isfinite().all():
            raise ValueError("RAEv2 predictor produced non-finite latents")
        return prediction

    @torch.no_grad()
    def sample_dino_latent(self, labels):
        labels = labels.to(self.device)
        batch_size = len(labels)
        noise = torch.randn(batch_size, *self.latent_size, device=self.device)
        context = labels
        if self.use_guidance:
            noise = torch.cat([noise, noise], dim=0)
            null = torch.full_like(labels, self.null_label)
            context = torch.cat([context, null], dim=0)
        model_kwargs = dict(context=context, attn_mask=None, **self.sample_model_kwargs)
        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            samples = self.sample_fn(noise, self.model_fn, **model_kwargs)[-1]
        if self.use_guidance:
            samples = samples.chunk(2, dim=0)[0]
        if samples.shape != (batch_size, *self.latent_size):
            raise ValueError(f"Unexpected RAEv2 latent shape: {samples.shape}")
        if not samples.isfinite().all():
            raise ValueError("RAEv2 sampler produced non-finite latents")
        return samples
