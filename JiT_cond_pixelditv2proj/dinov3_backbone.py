"""JiT grounding ablation 使用的本地 DINOv3-B/16 加载器。

The source tree is kept in the repository (under ``Ours_BaseT.../SSL_code``)
and the weights are resolved from the shared ``pretrain_models`` directory.
No network access is needed at model construction time.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from resource_paths import dinov3_repo, pretrain_models_root


PROJECT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PROJECT_ROOT.parent
_WEIGHT_FILENAMES = {
    "dinov3_vitb16": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    "dinov3_vits16": "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    "dinov3_vitl16": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
}


def normalize_model_name(name: str) -> str:
    aliases = {
        "dinov3": "dinov3_vitb16",
        "dinov3_vitb": "dinov3_vitb16",
        "dinov3_b": "dinov3_vitb16",
        "dinov3_vitb/16": "dinov3_vitb16",
    }
    return aliases.get(name.lower(), name.lower())


def _resolve_source(repo: Optional[str] = None) -> Path:
    # 优先使用脚本传入的目录，其次使用环境变量和仓库内已有的 DINOv3
    # 源码副本，整个过程不需要联网下载代码。
    candidates = []
    if repo:
        candidates.append(Path(repo).expanduser())
    for env_name in ("DINOV3_REPO", "DINOV3_REPO_DIR"):
        if os.environ.get(env_name):
            candidates.append(Path(os.environ[env_name]).expanduser())
    candidates.extend([
        dinov3_repo(),
        REPOSITORY_ROOT / "PixelDiTv2_ours_DiTlayer" / "third_party" / "dinov3",
        Path.home() / ".cache" / "torch" / "hub" / "facebookresearch_dinov3_main",
    ])
    for candidate in candidates:
        if (candidate / "hubconf.py").is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "DINOv3 source tree not found. Set DINOV3_REPO to a directory containing hubconf.py; "
        f"tried {candidates}"
    )


def _resolve_weights(model_name: str, weights_path: Optional[str] = None) -> Path:
    if weights_path:
        path = Path(os.path.expandvars(weights_path)).expanduser().resolve()
    else:
        # 权重统一从共享 pretrain_models 目录解析，避免依赖当前工作目录。
        root = pretrain_models_root()
        try:
            path = (root / _WEIGHT_FILENAMES[normalize_model_name(model_name)]).resolve()
        except KeyError as exc:
            raise ValueError(f"Unsupported DINOv3 model {model_name!r}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"DINOv3 weights not found: {path}")
    return path


def load_dino_v3(
    model_name: str = "dinov3_vitb16",
    weights_path: Optional[str] = None,
    repo: Optional[str] = None,
) -> nn.Module:
    """Construct a local DINOv3 model and load the complete checkpoint."""

    model_name = normalize_model_name(model_name)
    source = _resolve_source(repo)
    weights = _resolve_weights(model_name, weights_path)
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

    # 官方构造器在 pretrained=True 且 weights 为本地路径时会加载权重；
    # 对旧版 hubconf 保留显式 state_dict 加载作为兼容回退。
    from dinov3.hub import backbones

    constructor = getattr(backbones, model_name)
    try:
        model = constructor(pretrained=True, weights=str(weights))
    except TypeError:
        model = constructor(pretrained=False)
        state = torch.load(weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    return model
