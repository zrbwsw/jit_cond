"""PixelDiT2 training checkpoints. Frozen encoders and the RAEv2 DDT live externally."""
from contextlib import contextmanager
import hashlib
from pathlib import Path

import torch


FORMAT = "PixelDiTv2_cond_RAEv2/v1"
ARCH_FIELDS = ("model", "img_size", "class_num")
TRAIN_FIELDS = ("attn_dropout", "proj_dropout", "label_drop_prob", "t_mean_pixel",
                "t_std_pixel", "t_eps", "noise_scale_pixel", "ema_decay", "repa_weight",
                "repa_align_layer")


def latent_spec(args):
    path = Path(args.dino_stat_path).expanduser()
    return {
        "encoder": "dinov3l-k7",
        "shape": (1024, 16, 16),
        "stats_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "normalization": "raev2_multilayer_mean_plus_global_zscore_eps1e-5",
    }


def read_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != FORMAT:
        raise ValueError(f"Expected a {FORMAT} checkpoint: {path}")
    return checkpoint


def _trainable(model):
    return {name: p for name, p in model.named_parameters() if p.requires_grad}


@torch.no_grad()
def initialize_ema(model):
    model.ema_params = {name: p.detach().clone() for name, p in _trainable(model).items()}


def build_checkpoint(args, model, optimizer, epoch):
    if model.ema_params is None or model.ema_params.keys() != _trainable(model).keys():
        raise ValueError("EMA does not match the trainable parameter set")
    return {
        "format": FORMAT,
        "epoch": epoch,  # Number of completed epochs.
        "args": dict(vars(args)),
        "latent_spec": getattr(args, "latent_spec", None) or latent_spec(args),
        "model": model.net.state_dict(),
        "repa_projection": model.repa_projection.state_dict(),
        "model_ema": model.ema_params,
        "optimizer": optimizer.state_dict(),
    }


@torch.no_grad()
def resume_model(model, optimizer, checkpoint, args):
    if checkpoint.get("format") != FORMAT:
        raise ValueError(f"Expected {FORMAT} checkpoint")
    if checkpoint.get("latent_spec") != latent_spec(args):
        raise ValueError("DINOv3-L k=7 latent statistics differ from the Pixel checkpoint")
    fields = ARCH_FIELDS if optimizer is None else ARCH_FIELDS + TRAIN_FIELDS
    for field in fields:
        if checkpoint["args"][field] != getattr(args, field):
            raise ValueError(f"Checkpoint {field} differs from the requested configuration")
    model.net.load_state_dict(checkpoint["model"], strict=True)
    model.repa_projection.load_state_dict(checkpoint["repa_projection"], strict=True)
    if optimizer is None:
        model.repa_projection.requires_grad_(checkpoint["args"]["repa_weight"] > 0)
    trainable = _trainable(model)
    ema = checkpoint["model_ema"]
    if ema.keys() != trainable.keys():
        raise ValueError("EMA keys differ from the trainable parameter names")
    for name, parameter in trainable.items():
        if ema[name].shape != parameter.shape:
            raise ValueError(f"EMA shape mismatch for {name}")
    if optimizer is None:
        # Sampling-only: avoid retaining an extra H-sized copy of the weights on device.
        for name, parameter in trainable.items():
            parameter.copy_(ema[name])
        model.ema_loaded_for_sampling = True
    else:
        optimizer.load_state_dict(checkpoint["optimizer"])
        model.ema_params = {name: ema[name].to(parameter, copy=True)
                            for name, parameter in trainable.items()}
    args.start_epoch = checkpoint["epoch"]


@contextmanager
def use_ema(model):
    if getattr(model, "ema_loaded_for_sampling", False):
        yield
        return
    trainable = _trainable(model)
    if model.ema_params is None or model.ema_params.keys() != trainable.keys():
        raise ValueError("EMA is uninitialized or mismatched")
    # Swap trainable and EMA storage; online evaluation needs no third copy of PixelDiT2-H.
    # Keep Parameter objects (and hence optimizer / DDP references) stable.
    swapped = []
    with torch.no_grad():
        try:
            for name, parameter in trainable.items():
                original = parameter.detach()
                parameter.data = model.ema_params[name]
                model.ema_params[name] = original
                swapped.append(name)
            yield
        finally:
            for name in swapped:
                parameter = trainable[name]
                averaged = parameter.detach()
                parameter.data = model.ema_params[name]
                model.ema_params[name] = averaged
