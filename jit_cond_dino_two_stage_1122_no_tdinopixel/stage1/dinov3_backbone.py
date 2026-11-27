"""本地 DINOv3 加载器，共享 resource_paths 的路径解析。

权重和 DINOv3 源码目录通过显式参数或环境变量解析，全程不联网。
保留 join-dino 的 backbone 构造与权重格式。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from resource_paths import dinov3_repo, pretrain_models_root


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
        "dinov3b": "dinov3_vitb16",
        "dinov3_vitb/16": "dinov3_vitb16",
    }
    return aliases.get(name.lower(), name.lower())


def _pretrain_root() -> Path:
    return pretrain_models_root()


def _resolve_source(repo: Optional[str] = None) -> Path:
    candidates = []
    if repo:
        candidates.append(Path(repo).expanduser())
    for env_name in ("DINOV3_REPO", "DINOV3_REPO_DIR"):
        if os.environ.get(env_name):
            candidates.append(Path(os.environ[env_name]).expanduser())
    candidates.append(Path.home() / ".cache" / "torch" / "hub" / "facebookresearch_dinov3_main")
    candidates.append(dinov3_repo())
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
        try:
            path = (_pretrain_root() / _WEIGHT_FILENAMES[normalize_model_name(model_name)]).resolve()
        except KeyError as exc:
            raise ValueError(f"Unsupported DINOv3 model {model_name!r}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"DINOv3 weights not found: {path}")
    return path


def load_dino_v3(model_name: str = "dinov3_vitb16", weights_path: Optional[str] = None,
                 repo: Optional[str] = None) -> nn.Module:
    """构造本地 DINOv3 并加载完整 checkpoint。"""
    model_name = normalize_model_name(model_name)
    source = _resolve_source(repo)
    weights = _resolve_weights(model_name, weights_path)
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

    from dinov3.hub import backbones
    constructor = getattr(backbones, model_name)
    try:
        model = constructor(pretrained=True, weights=str(weights))
    except TypeError:
        model = constructor(pretrained=False)
        state = torch.load(weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    return model
