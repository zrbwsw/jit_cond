"""解析 JiT_cond_pixelditv2proj 实验使用的本地共享资源。

The launchers may run from either the repository root or ``JiT_cond``.  Keep
all defaults in one place so data, checkpoints, and FID statistics do not
depend on the current working directory.  Every value can be overridden with
an environment variable, which is useful on a remote GPU worker.
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PROJECT_ROOT.parent
DEFAULT_IMAGENET_ROOT = Path("/mnt/shared-storage-user/datafactory-share/gushishen/imagenet")
DINOV3_REPO_CANDIDATES = (
    REPOSITORY_ROOT / "5-3-Residual(0,1)-1017(DINOv3-B16)" / "SSL_code" / "dinov3",
    REPOSITORY_ROOT / "Ours_BaseT_rmsnorm2_Residual_1016_normft_1028" / "SSL_code" / "dinov3",
    REPOSITORY_ROOT / "Our_rmsnorm2_4DiT_Residual_ft_1024" / "SSL_code" / "dinov3",
)
DINOV3_B16_WEIGHTS = "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"


def _resolve(value: str | os.PathLike[str]) -> Path:
    """Expand a user path and make it absolute without requiring it to exist."""

    return Path(value).expanduser().resolve(strict=False)


def _env_path(name: str, default: str | os.PathLike[str]) -> Path:
    return _resolve(os.environ.get(name, str(default)))


def imagenet_root(value: str | os.PathLike[str] | None = None) -> Path:
    """Return the ImageNet root containing ``train/`` and ``val/``."""

    return _resolve(value) if value is not None else _env_path("IMAGENET_ROOT", DEFAULT_IMAGENET_ROOT)


def pretrain_models_root(value: str | os.PathLike[str] | None = None) -> Path:
    default = REPOSITORY_ROOT / "pretrain_models"
    if not default.is_dir() and Path("/data0/pretrain_models").is_dir():
        default = Path("/data0/pretrain_models")
    return _resolve(value) if value is not None else _env_path("PRETRAIN_MODELS_ROOT", default)


def dinov3_repo(value: str | os.PathLike[str] | None = None) -> Path:
    """解析 torch.hub 使用的本地 DINOv3 源码目录。"""

    if value is not None:
        return _resolve(value)
    for env_name in ("DINOV3_REPO", "DINOV3_REPO_DIR"):
        override = os.environ.get(env_name)
        if override:
            return _resolve(override)
    for candidate in DINOV3_REPO_CANDIDATES:
        if (candidate / "hubconf.py").is_file():
            return candidate.resolve()
    # Return the canonical candidate to keep a useful error path for callers.
    return DINOV3_REPO_CANDIDATES[0].resolve()


def dinov3_weights_path(value: str | os.PathLike[str] | None = None) -> Path:
    """解析 DINOv3-B/16 权重，并检查标准本地文件名是否存在。"""

    path = _resolve(value) if value is not None else pretrain_models_root() / DINOV3_B16_WEIGHTS
    if not path.is_file():
        raise FileNotFoundError(
            f"DINOv3-B/16 weights not found at {path}; set PRETRAIN_MODELS_ROOT or pass an explicit path"
        )
    return path


def fid_stats_root(value: str | os.PathLike[str] | None = None) -> Path:
    return _resolve(value) if value is not None else _env_path("FID_STATS_ROOT", REPOSITORY_ROOT / "fid_stats")


def fid_stats_path(img_size: int, root: str | os.PathLike[str] | None = None) -> Path:
    """返回指定分辨率的 JiT FID 统计文件。

    Raises ``ValueError`` for an unsupported size and ``FileNotFoundError``
    when the expected local statistics are not present.
    """

    names = {256: "jit_in256_stats.npz", 512: "jit_in512_stats.npz"}
    try:
        filename = names[int(img_size)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"JiT FID statistics are available only for 256 or 512, got {img_size!r}") from exc
    path = fid_stats_root(root) / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"FID statistics not found at {path}; set FID_STATS_ROOT to a directory containing {filename}"
        )
    return path


def validate_imagenet_root(root: str | os.PathLike[str] | None = None) -> Path:
    """Validate the ImageFolder layout needed by training and return its root."""

    path = imagenet_root(root)
    missing = [name for name in ("train", "val") if not (path / name).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"ImageNet root {path} is missing {'/'.join(missing)}; expected train/ and val/ directories"
        )
    return path

