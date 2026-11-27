"""Evaluate noised Pixel JiT with training-matched real-image grounding."""
import argparse
import json
from functools import partial
from pathlib import Path

import torch

from checkpoint_io import latent_spec, read_checkpoint, resume_model
from denoiser import Denoiser
from engine_jit import evaluate
from grounding_evaluation import RealGroundingSampler, ValGroundingProvider
from main_jit import get_args_parser, validate_args
from stage1.denoiser_dino import sample_logit_normal
from util.training import initialize_run, make_log_writer


class NoisedRealGroundingSampler(RealGroundingSampler):
    def __init__(self, model, provider, *, num_images, batch_size, world_size=1, rank=0,
                 fixed_t_dino_pixel=None):
        if fixed_t_dino_pixel is not None and not 0 <= fixed_t_dino_pixel <= 1:
            raise ValueError("fixed_t_dino_pixel must be in [0, 1]")
        self.fixed_t_dino_pixel = fixed_t_dino_pixel
        super().__init__(
            model, provider, False,
            num_images=num_images, batch_size=batch_size,
            world_size=world_size, rank=rank,
        )

    @torch.no_grad()
    def __call__(self, labels):
        clean = super().__call__(labels)
        if self.fixed_t_dino_pixel is not None:
            t_dino_pixel = torch.full(
                (len(labels),), self.fixed_t_dino_pixel, device=labels.device)
        elif self.model.t_schedule_dino_pixel == "uniform":
            beta = self.model.noise_beta_dino_pixel
            t_dino_pixel = 1 - beta + beta * torch.rand(len(labels), device=labels.device)
        else:
            t_dino_pixel = sample_logit_normal(
                len(labels), labels.device,
                self.model.t_mean_dino, self.model.t_std_dino,
            )
        t_cond = t_dino_pixel.view(-1, 1, 1, 1)
        z_cond = (
            t_cond * clean
            + (1 - t_cond) * torch.randn_like(clean) * self.model.noise_scale_dino
        )
        if self.model.condition_mode_pixel == "pred":
            z_cond = self.model.dino_net(z_cond, t_dino_pixel, labels)
        return z_cond


def generate_noised_grounding(model, sampler, generate_fn, labels):
    z_cond = sampler(labels)
    return generate_fn(
        labels,
        grounding_tokens=model._as_tokens(z_cond),
    )


def main(args):
    if args.condition_mode_pixel not in ("noised", "pred"):
        raise ValueError("real-image grounding evaluation requires --condition_mode_pixel noised or pred")
    checkpoint_path = validate_args(args)
    args.latent_spec = latent_spec(args)
    device, seed = initialize_run(args)
    model = Denoiser(args).to(device)
    resume_model(model, None, read_checkpoint(checkpoint_path), args)
    model.eval()

    provider = ValGroundingProvider(
        args.data_path, args.img_size, args.class_num, args.grounding_split,
        per_class=args.grounding_per_class, selection_seed=args.grounding_selection_seed)
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    if not distributed or torch.distributed.get_rank() == 0:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "grounding_selection.json"
        manifest_path.write_text(json.dumps(provider.selection_record(), indent=2) + "\n")
        print(f"Grounding selection manifest: {manifest_path}")
    sampler = NoisedRealGroundingSampler(
        model, provider,
        num_images=args.num_images, batch_size=args.gen_bsz,
        world_size=torch.distributed.get_world_size() if distributed else 1,
        rank=torch.distributed.get_rank() if distributed else 0,
        fixed_t_dino_pixel=args.fixed_t_dino_pixel,
    )
    model.generate = partial(generate_noised_grounding, model, sampler, model.generate)
    model.method_dino = f"real{args.grounding_split}-{args.condition_mode_pixel}"
    if args.fixed_t_dino_pixel is not None:
        model.method_dino += f"-{args.fixed_t_dino_pixel}"
    model.steps_dino = 1 if args.condition_mode_pixel == "pred" else 0
    model.cfg_dino = 1.0
    model.cfg_interval_dino = (0.0, 0.0)
    noise_description = (
        f"schedule={model.t_schedule_dino_pixel}, beta={model.noise_beta_dino_pixel}"
        if args.fixed_t_dino_pixel is None
        else f"fixed t_dino_pixel={args.fixed_t_dino_pixel}"
    )
    dino_description = "DinoDiT one-step prediction" if args.condition_mode_pixel == "pred" else "DinoDiT bypassed"
    print(
        f"Grounding: real {args.grounding_split} images with training-matched noise; "
        f"{noise_description}, noise_scale={model.noise_scale_dino}; {dino_description}"
    )
    print(f"Grounding selection: per_class={args.grounding_per_class}, seed={args.grounding_selection_seed}")
    print(f"Loaded Pixel checkpoint: {checkpoint_path}; epoch={args.start_epoch}")

    log_writer = None if args.skip_metrics else make_log_writer(args.output_dir)
    try:
        devices = [torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            evaluate(model, args, args.start_epoch, batch_size=args.gen_bsz, log_writer=log_writer)
    finally:
        if log_writer is not None:
            log_writer.close()


if __name__ == "__main__":
    parser = get_args_parser()
    parser.add_argument("--grounding_split", default="val")
    parser.add_argument("--grounding_per_class", type=int, default=None)
    parser.add_argument("--grounding_selection_seed", type=int, default=0)
    parser.add_argument("--fixed_t_dino_pixel", type=float, default=None)
    parser.add_argument("--keep_images", action="store_true")
    parser.add_argument("--skip_metrics", action="store_true")
    parser.set_defaults(evaluate_gen=True)
    main(parser.parse_args())
