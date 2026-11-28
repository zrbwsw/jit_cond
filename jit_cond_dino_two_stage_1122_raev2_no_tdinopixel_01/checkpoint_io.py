"""Checkpoint loading and EMA utilities for RAEv2-grounded Pixel JiT."""
import hashlib
from contextlib import contextmanager
from pathlib import Path

import torch


def latent_spec(args):
    path = Path(args.dino_stat_path).expanduser()
    return {
        "encoder": "dinov3l-k7",
        "stats_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "normalization": "raev2_multilayer_mean_plus_global_zscore_eps1e-5",
    }


def read_checkpoint(path):
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("Expected a checkpoint containing 'model'")
    return checkpoint


def _saved_args(checkpoint):
    saved = checkpoint.get("args", {})
    return saved if isinstance(saved, dict) else vars(saved)


def validate_metadata(checkpoint, args, *, resume=False):
    saved = _saved_args(checkpoint)
    fields = ["model", "img_size", "class_num"]
    if resume:
        fields += [
            "t_mean_pixel", "t_std_pixel", "noise_scale_pixel", "label_drop_prob",
            "ema_decay1", "ema_decay2", "attn_dropout", "proj_dropout",
            "condition_mode_pixel", "noise_beta_dino_pixel",
        ]
    legacy_defaults = {"condition_mode_pixel": "clean", "noise_beta_dino_pixel": 0.25}
    for key in fields:
        if key not in saved and key not in legacy_defaults:
            raise ValueError(f"Checkpoint lacks {key} metadata")
        saved_value = saved.get(key, legacy_defaults.get(key))
        if saved_value != getattr(args, key):
            raise ValueError(f"Checkpoint {key}={saved_value!r}, requested {getattr(args, key)!r}")
    if "latent_spec" in checkpoint and checkpoint["latent_spec"] != latent_spec(args):
        raise ValueError("DINOv3-L k7 latent statistics do not match the checkpoint")


def load_branch(model, checkpoint, branch, weights, args):
    validate_metadata(checkpoint, args)
    module = getattr(model, branch)
    prefix = branch + "."
    state = {key[len(prefix):]: value for key, value in checkpoint["model"].items() if key.startswith(prefix)}
    if not state:
        raise ValueError(f"Checkpoint does not contain {prefix} weights")
    if weights != "model":
        ema = checkpoint.get("model_" + weights, {})
        required = [name for name, parameter in module.named_parameters() if parameter.requires_grad]
        missing = [prefix + name for name in required if prefix + name not in ema]
        if missing:
            raise ValueError(f"Requested {weights} is missing {branch} parameters: {missing[:3]}")
        state.update({key[len(prefix):]: value for key, value in ema.items() if key.startswith(prefix)})
    module.load_state_dict(state, strict=True)
    print(f"Initialized {branch} from {weights}; optimizer and epoch are NOT restored")


def resume_model(model, optimizer, checkpoint, args):
    if checkpoint.get("training_stage") != args.training_stage or checkpoint.get("format_version") != 1:
        raise ValueError("Cannot resume a checkpoint from another training stage or format")
    validate_metadata(checkpoint, args, resume=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    for index in (1, 2):
        ema = checkpoint[f"model_ema{index}"]
        if set(ema) != {name for name, _ in trainable}:
            raise ValueError("EMA parameter names do not match this training stage")
        setattr(model, f"ema_params{index}", [ema[name].to(parameter).clone() for name, parameter in trainable])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    args.start_epoch = checkpoint["epoch"]
    print(f"Resumed Pixel JiT checkpoint at epoch {args.start_epoch}")


@torch.no_grad()
def initialize_ema(model):
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    model.ema_params1 = [parameter.clone() for parameter in params]
    model.ema_params2 = [parameter.clone() for parameter in params]
    print("Initialized EMA parameters")


@contextmanager
def use_ema(model):
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if model.ema_params1 is None or len(params) != len(model.ema_params1):
        raise ValueError("EMA1 does not match trainable parameters")
    with torch.no_grad():
        original = [parameter.clone() for parameter in params]
        for parameter, ema in zip(params, model.ema_params1):
            parameter.copy_(ema)
    try:
        yield
    finally:
        with torch.no_grad():
            for parameter, saved in zip(params, original):
                parameter.copy_(saved)
