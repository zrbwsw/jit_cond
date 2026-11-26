"""Locate the shared assets on the workstation or the H200 cluster."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
REPO = PROJECT_ROOT.parent
WORK = REPO.parent
STATS_NAME = "dinov3l_k7_imagenet1k_stats.pt"


def first_available(paths):
    return next((path for path in paths if path.exists()), paths[0])


def asset(env, *paths):
    override = os.environ.get(env)
    return Path(override).expanduser() if override else first_available(paths)


def shared_file(env, root_env, roots, relative):
    """Resolve each file independently when weights are split between shared directories."""
    if os.environ.get(root_env):
        roots = [Path(os.environ[root_env]).expanduser()]
    return asset(env, *(root / relative for root in roots))


PIXEL_ROOT = asset("PIXEL_DIFFUSION_ROOT", WORK / "Pixel_Diffusion", WORK / "Pixel_DIffusion")
RAEV2_ROOT = asset("RAEV2_ROOT", PIXEL_ROOT / "RAEv2", WORK / "RAEv2")


def asset_defaults():
    pretrain = [Path("/data0/pretrain_models"), REPO / "pretrain_models", PIXEL_ROOT / "pretrain_models"]
    rae_weights = [RAEV2_ROOT / "pretrained_models", WORK / "RAEv2"]
    stats = shared_file("DINO_STAT_PATH", "RAEV2_MODELS_ROOT", rae_weights,
                        "stage1/imagenet/dinov3l-k7/stats.pt")
    return dict(
        data_path=str(asset("IMAGENET_ROOT", Path("/data0/datasets/imagenet"), WORK.parent / "imagenet")),
        dino_stat_path=str(asset("DINO_STAT_PATH", stats,
                                 REPO / "jit_cond_dino_two_stage_1122_raev2/stat" / STATS_NAME)),
        dino_encoder_weights_path=str(shared_file("DINO_ENCODER_WEIGHTS_PATH", "PRETRAIN_MODELS_ROOT",
                                      pretrain, "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")),
        dino_encoder_repo=str(asset("DINOV3_REPO", PIXEL_ROOT / "Ours_BaseT_rmsnorm2_Residual_1016/SSL_code/dinov3",
                                    Path.home() / ".cache/torch/hub/facebookresearch_dinov3_main")),
        repa_encoder_weights_path=str(shared_file("REPA_ENCODER_WEIGHTS_PATH", "PRETRAIN_MODELS_ROOT",
                                      pretrain, "dinov2_vitb14_pretrain.pth")),
        raev2_root=str(RAEV2_ROOT),
        raev2_config=str(asset("RAEV2_CONFIG", RAEV2_ROOT / "configs/stage2/sampling/imagenet-dinov3l-k7.yaml")),
        raev2_checkpoint=str(shared_file("RAEV2_CHECKPOINT", "RAEV2_MODELS_ROOT", rae_weights,
                                         "stage2/imagenet/dinov3l-k7/checkpoint.pt")),
        fid_stats=str(shared_file("FID_STATS_PATH", "FID_STATS_ROOT", [REPO / "fid_stats", PIXEL_ROOT / "fid_stats"],
                                  "jit_in256_stats.npz")),
    )
