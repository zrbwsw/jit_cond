"""Evaluate Pixel JiT with clean real-image DINO grounding."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from checkpoint_io import latent_spec, read_checkpoint, resume_model
from denoiser import Denoiser
from engine_jit import evaluate
from main_jit import get_args_parser, validate_args
from util.crop import center_crop_arr
from util.training import initialize_run, make_log_writer


class ValGroundingProvider:
    def __init__(self, root, img_size, num_classes, split="val", per_class=None, selection_seed=0):
        self.root = Path(root)
        self.img_size = img_size
        self.num_classes = num_classes
        self.split = split
        self.per_class = per_class
        self.selection_seed = selection_seed
        if per_class is not None and per_class < 1:
            raise ValueError("grounding_per_class must be positive")
        split_dir = self.root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Grounding split not found: {split_dir}")
        self.classes = sorted(path.name for path in split_dir.iterdir() if path.is_dir())
        if len(self.classes) != num_classes:
            raise ValueError(f"Found {len(self.classes)} classes in {split_dir}, expected {num_classes}")
        self.files = []
        rng = np.random.default_rng(selection_seed)
        for name in self.classes:
            files = sorted(path.name for path in (split_dir / name).iterdir()
                           if path.suffix.lower() in {".jpeg", ".jpg", ".png", ".bmp", ".webp"})
            if not files:
                raise FileNotFoundError(f"No grounding images in {split_dir / name}")
            if per_class is not None:
                if len(files) < per_class:
                    raise ValueError(f"Found {len(files)} images in {split_dir / name}, need {per_class}")
                indices = rng.choice(len(files), size=per_class, replace=False)
                files = sorted(files[index] for index in indices)
            self.files.append(files)

    def selection_record(self):
        return {
            "split": self.split,
            "per_class": self.per_class if self.per_class is not None else len(self.files[0]),
            "selection_seed": self.selection_seed if self.per_class is not None else None,
            "files": dict(zip(self.classes, self.files)),
        }

    def load_batch(self, labels, intra_indices):
        images = []
        for label, index in zip(labels, intra_indices):
            name = self.classes[int(label)]
            filename = self.files[int(label)][int(index) % len(self.files[int(label)])]
            with Image.open(self.root / self.split / name / filename) as image:
                array = np.array(center_crop_arr(image.convert("RGB"), self.img_size))
            images.append(torch.from_numpy(array).permute(2, 0, 1))
        return torch.stack(images).float().div_(255.0).mul_(2.0).sub_(1.0)


class RealGroundingSampler:
    def __init__(self, model, provider, match_dino_noise_rng=True, *,
                 num_images, batch_size, world_size=1, rank=0):
        if num_images < 1 or num_images % provider.num_classes:
            raise ValueError("num_images must be positive and divisible by the number of classes")
        if batch_size < 1 or world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("Invalid generation batch or rank configuration")
        self.model = model
        self.provider = provider
        self.match_dino_noise_rng = match_dino_noise_rng
        self.num_images = num_images
        self.per_class = num_images // provider.num_classes
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.call_index = 0

    @torch.no_grad()
    def __call__(self, labels):
        labels_np = labels.detach().cpu().numpy()
        start = self.call_index * self.world_size * self.batch_size + self.rank * self.batch_size
        generation_indices = np.arange(start, start + len(labels_np))
        generation_indices = np.minimum(generation_indices, self.num_images - 1)
        intra_indices = generation_indices % self.per_class
        self.call_index += 1
        images = self.provider.load_batch(labels_np, intra_indices).to(labels.device)
        clean = self.model.encode_grounding(images)
        if self.match_dino_noise_rng:
            noise_shape = (labels.size(0), self.model.dino_channels,
                           self.model.latent_size, self.model.latent_size)
            torch.randn(noise_shape, device=labels.device)
        return clean


def main(args):
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
    model.sample_dino_latent = RealGroundingSampler(
        model, provider, args.match_dino_noise_rng,
        num_images=args.num_images, batch_size=args.gen_bsz,
        world_size=torch.distributed.get_world_size() if distributed else 1,
        rank=torch.distributed.get_rank() if distributed else 0)
    model.method_dino = f"real{args.grounding_split}"
    model.steps_dino = 0
    model.cfg_dino = 1.0
    model.cfg_interval_dino = (0.0, 0.0)
    print(f"Grounding upper bound: real {args.grounding_split} clean images, t_dino_pixel=1; DinoDiT bypassed")
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
    parser.add_argument("--match_dino_noise_rng", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip_metrics", action="store_true")
    parser.set_defaults(evaluate_gen=True)
    main(parser.parse_args())
