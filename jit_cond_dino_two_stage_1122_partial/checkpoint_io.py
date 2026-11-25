"""Named branch loading: never resume a joint optimizer into a split trainer."""

import hashlib
from contextlib import contextmanager
from pathlib import Path

import torch

from stage1.dinov2_encoder import latent_stat_path


LEGACY_KEYS = {
    "noise_scale_dino": "noise_scale",
    "noise_scale_pixel": "noise_scale",
    "t_mean_dino": "D_mean",
    "t_std_dino": "D_std",
    "t_mean_pixel": "P_mean",
    "t_std_pixel": "P_std",
    "loss_scale_dino": "dino_loss_scale",
    "noise_beta_dino_pixel": "dino_noise_beta",
    "t_schedule_dino_pixel": "condition_time_schedule",
}


def latent_spec(args):
    path = latent_stat_path(args.dino_encoder, args.dino_stat_path)
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
    fields = ["dino_encoder", "dino_model", "img_size", "grounding_dim", "class_num", "noise_scale_dino", "t_eps"]
    if pixel:
        fields += ["model"]
    if resume and not getattr(args, "evaluate_gen", False):
        fields += ["t_mean_dino", "t_std_dino", "label_drop_prob", "ema_decay1", "ema_decay2",
                   "attn_dropout", "proj_dropout"]
        if pixel:
            fields += ["condition_mode_pixel", "t_schedule_dino_pixel", "noise_beta_dino_pixel",
                       "noise_scale_pixel", "t_mean_pixel", "t_std_pixel"]
            # P2/P3: logit_normal 分支专用参数一并校验，确保续训分布与初次训练一致。
            # 旧 checkpoint 若缺失这两个键则注入默认值（与 argparse 默认值相同），
            # 以便对旧实验 checkpoint 透明续训而不报错。
            # partial_trajectory 是新增模式，旧 checkpoint 默认为 pred。
            saved = {
                "t_mean_dino_pixel": 1.95,
                "t_std_dino_pixel": 0.5,
                "condition_mode_pixel": "pred",
                "t_part": 0.6,
                "steps_part": 10,
                **saved,
            }
            fields += ["t_mean_dino_pixel", "t_std_dino_pixel"]
            # ===== 新增：partial_trajectory 参数校验 =====
            if saved.get("condition_mode_pixel") == "partial_trajectory":
                fields += ["t_part", "steps_part"]
            # =====
        else:
            fields += ["loss_scale_dino", "recon_weight", "recon_train_steps", "recon_train_batch",
                       "sampling_method_dino", "cfg_dino", "interval_min_dino", "interval_max_dino"]
    for key in fields:
        saved_key = key if key in saved else LEGACY_KEYS.get(key, key)
        if saved_key not in saved:
            raise ValueError(f"Checkpoint lacks {key} metadata; verify its latent/model configuration first")
        if saved[saved_key] != getattr(args, key):
            raise ValueError(f"Checkpoint {key}={saved[saved_key]!r}, requested {getattr(args, key)!r}")
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
    if not getattr(args, "evaluate_gen", False):
        optimizer.load_state_dict(checkpoint["optimizer"])
    if args.training_stage == "pixel" and getattr(args, "condition_mode_pixel", "") == "cached_recon":
        args.expected_cache_id = _saved_args(checkpoint).get("cache_id")
        if not args.expected_cache_id and not getattr(args, "evaluate_gen", False):
            raise ValueError("Cached training checkpoint lacks cache_id")
    args.start_epoch = checkpoint["epoch"]
    # Preserve provenance rather than overwriting it with empty CLI defaults on resume.
    saved = _saved_args(checkpoint)
    for key in ("dino_checkpoint", "dino_weights", "init_pixel", "init_pixel_weights"):
        if key in saved and hasattr(args, key):
            setattr(args, key, saved[key])


def completed_epochs(checkpoint):
    # 旧 JoinDino 文件可能保存零基索引。新版 checkpoint 保存已完成轮数。
    epoch = int(checkpoint["epoch"])
    if checkpoint.get("format_version") != 1 and "save_interval" not in _saved_args(checkpoint):
        epoch += 1
    return epoch
