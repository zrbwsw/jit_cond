from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from checkpoint_io import latent_spec, read_checkpoint, resume_model
from denoiser import Denoiser
from main_jit import get_args_parser
from util.recon_cache import cached_dataset, digest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = PROJECT_ROOT / "checkpoints/stage1-dinov3b-dinodit-b-recon-noflip-e80-stage2-pixeljit-b-dinodit-cached-recon-perfile-e320/checkpoint-last-perfile.pth"
CACHE = Path("/mnt/shared-storage-user/datafactory-share/gushishen/imagenet/stage1-dinov3b-dinodit-b-recon-noflip-e80")

args = get_args_parser().parse_args([
    "--resume", str(CHECKPOINT),
    "--dino_weights", "ema1",
    "--dino_stat_path", "/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/jit_cond_ablation/JiT_cond_join_dino/stat/dinov3b_rae_imagenet1k_stats.pt",
    "--grounding_cache", str(CACHE),
    "--cache_cfg_dino", "1.0", "--cfg_dino", "2.0",
    "--data_path", "/mnt/shared-storage-user/datafactory-share/gushishen/imagenet",
    "--condition_mode_pixel", "cached_recon",
    "--steps_dino", "25", "--sampling_method_dino", "heun",
    "--device", "cuda:0",
])
args.latent_spec = latent_spec(args)
args.evaluate_gen = True
model = Denoiser(args)
resume_model(model, None, read_checkpoint(CHECKPOINT), args)
model.to(args.device).eval()
dataset = cached_dataset(args, model)
manifest = __import__("json").loads((CACHE / "manifest.json").read_text())
assert manifest["id"] == digest([manifest["spec"], manifest["count"]])
assert args.start_epoch == 45, args.start_epoch
assert args.cache_id == manifest["id"] == args.expected_cache_id
assert len(dataset) == 1281167
for index in (0, len(dataset) - 1):
    image, label, feature = dataset[index]
    assert image.shape == (3, 256, 256)
    assert feature.shape == (256, 768) and feature.dtype == torch.float16
    assert torch.isfinite(feature).all()
    assert label == dataset.folder.samples[index][1]
print(f"PASS: per-file cache, EMA1 teacher, cache_id={args.cache_id}, resume_epoch={args.start_epoch}")
print("First and last images, labels, and per-file latent rows are valid")
