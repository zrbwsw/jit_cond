"""Named branch loading: never resume a joint optimizer into a split trainer."""

import hashlib
from contextlib import contextmanager
from pathlib import Path

import torch

from latent_force.dinov2_encoder import ENCODER_PRESETS, STAT_DIR


def latent_spec(args):
    path = Path(args.dino_stat_path or Path(STAT_DIR) / ENCODER_PRESETS[args.dino_encoder]["stat"])
    return {
        "encoder": args.dino_encoder,
        "stats_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "normalization": "no_affine_mean_var_eps1e-5_no_extra_scale",
    }


def read_checkpoint(path):
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    # Only load trusted local experiment checkpoints (they contain argparse.Namespace).
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("Expected a join-dino or independent checkpoint containing 'model'")
    return checkpoint


def _saved_args(checkpoint):
    saved = checkpoint.get("args", {})
    return saved if isinstance(saved, dict) else vars(saved)


def validate_metadata(checkpoint, args, *, pixel=False, resume=False):
    saved = _saved_args(checkpoint)
    fields = ["dino_encoder", "dino_model", "img_size", "grounding_dim", "class_num", "noise_scale", "t_eps"]
    if pixel:
        fields += ["model"]
    if resume and not args.evaluate_gen:
        fields += ["condition_time_schedule", "dino_noise_beta", "P_mean", "P_std", "D_mean", "D_std",
                   "label_drop_prob", "ema_decay1", "ema_decay2", "dino_loss_scale",
                   "attn_dropout", "proj_dropout"]
    for key in fields:
        if key not in saved:
            raise ValueError(f"Checkpoint lacks {key} metadata; verify its latent/model configuration first")
        if saved[key] != getattr(args, key):
            raise ValueError(f"Checkpoint {key}={saved[key]!r}, requested {getattr(args, key)!r}")
    if "latent_spec" in checkpoint and checkpoint["latent_spec"] != latent_spec(args):
        raise ValueError("DINO latent normalization/statistics do not match the checkpoint")


def load_branch(model, checkpoint, branch, weights, args):
    validate_metadata(checkpoint, args, pixel=(branch == "net"))
    module = getattr(model, branch)
    prefix = branch + "."
    state = {k[len(prefix):]: v for k, v in checkpoint["model"].items() if k.startswith(prefix)}
    if not state:
        raise ValueError(f"Checkpoint does not contain {prefix} weights")
    if weights != "model":
        ema = checkpoint.get("model_" + weights, {})
        required = (model.dino_trainable_names if branch == "dino_net" else
                    [n for n, p in module.named_parameters() if p.requires_grad])
        missing = [prefix + n for n in required if prefix + n not in ema]
        if missing:
            raise ValueError(f"Requested {weights} is missing {branch} parameters: {missing[:3]}")
        state.update({k[len(prefix):]: v for k, v in ema.items() if k.startswith(prefix)})
    module.load_state_dict(state, strict=True)
    if branch == "dino_net":
        # The fixed encoder must be the one used to train this latent diffusion model.
        prefix = "dino_encoder."
        encoder_state = {k[len(prefix):]: v for k, v in checkpoint["model"].items() if k.startswith(prefix)}
        model.dino_encoder.load_state_dict(encoder_state, strict=True)
        if "latent_spec" not in checkpoint:
            print("Legacy checkpoint: verify the original DINO statistics match the selected stat file.")
    print(f"Initialized {branch} from {weights}; optimizer and epoch are NOT restored")


@torch.no_grad()
def initialize_ema(model):
    params = [p for p in model.parameters() if p.requires_grad]
    model.ema_params1 = [p.clone() for p in params]
    model.ema_params2 = [p.clone() for p in params]


@contextmanager
def use_ema(model):
    params = [p for p in model.parameters() if p.requires_grad]
    if model.ema_params1 is None or len(params) != len(model.ema_params1):
        raise ValueError('EMA1 does not match trainable parameters')
    with torch.no_grad():
        original = [p.clone() for p in params]
        for param, ema in zip(params, model.ema_params1):
            param.copy_(ema)
    try:
        yield
    finally:
        with torch.no_grad():
            for param, saved in zip(params, original):
                param.copy_(saved)


def resume_model(model, optimizer, checkpoint, args):
    if checkpoint.get("training_stage") != args.training_stage or checkpoint.get("format_version") != 1:
        raise ValueError("Cannot resume a joint/other-stage checkpoint; use --dino_checkpoint or --init_pixel instead")
    validate_metadata(checkpoint, args, pixel=(args.training_stage == "pixel"), resume=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    for index in (1, 2):
        ema = checkpoint[f"model_ema{index}"]
        if set(ema) != {n for n, _ in trainable}:
            raise ValueError("EMA parameter names do not match this training stage")
        setattr(model, f"ema_params{index}", [ema[n].to(p).clone() for n, p in trainable])
    if not args.evaluate_gen:
        optimizer.load_state_dict(checkpoint["optimizer"])
    args.start_epoch = checkpoint["epoch"]
    # Preserve provenance rather than overwriting it with empty CLI defaults on resume.
    for key in ("dino_checkpoint", "dino_weights", "init_pixel", "init_pixel_weights"):
        setattr(args, key, _saved_args(checkpoint).get(key, getattr(args, key)))
