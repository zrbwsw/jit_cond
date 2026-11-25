"""Per-file grounding cache (version 3).

每张图对应一个独立的 .npy 文件，路径结构与 ImageNet 镜像：
  <cache_root>/<synset>/<stem>.npy   shape=(256,768) dtype=float16

相比版本 2 的分片方式，全局 shuffle 后每个 worker 打开独立小文件，
OS page cache 自然命中，彻底消除分片 lru_cache 失效和多 worker I/O 竞争。
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import functional as TF

from util.crop import center_crop_arr


# ---------------------------------------------------------------------------
# hashing helpers（与版本 2 保持兼容）
# ---------------------------------------------------------------------------

def digest(data):
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def teacher_hash(model):
    h = hashlib.sha256()
    for scope in ("dino_net", "dino_encoder"):
        for key, value in sorted(getattr(model, scope).state_dict().items()):
            tensor = value.detach().to("cpu").contiguous()
            h.update(json.dumps([scope, key, str(tensor.dtype), list(tensor.shape)]).encode())
            h.update(tensor.reshape(-1).view(torch.uint8).numpy())
    return h.hexdigest()


def dataset_hash(folder):
    h = hashlib.sha256(json.dumps(folder.class_to_idx, sort_keys=True).encode())
    for path, label in folder.samples:
        key = Path(path).relative_to(folder.root).as_posix()
        h.update(json.dumps([key, label], ensure_ascii=True).encode() + b"\n")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# per-file path convention
# ---------------------------------------------------------------------------

def feature_path(cache_root, image_path, image_root):
    """Mirror ImageNet path under cache_root, replacing extension with .npy."""
    rel = Path(image_path).relative_to(image_root)
    return Path(cache_root) / rel.with_suffix(".npy")


def image_view(folder, idx, size):
    path, label = folder.samples[idx]
    image = center_crop_arr(folder.loader(path), size)
    return TF.pil_to_tensor(image), label


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CachedFolder(torch.utils.data.Dataset):
    """每个 __getitem__ 只打开一个 384 KB 文件，与全局 shuffle 完全兼容。"""

    def __init__(self, folder, cache_root, spec):
        self.folder = folder
        self.cache_root = Path(cache_root)
        self.spec = spec
        self._image_root = Path(folder.root)

    def __len__(self):
        return len(self.folder)

    def __getitem__(self, idx):
        s = self.spec
        image, label = image_view(self.folder, idx, s["size"])
        image_path, _ = self.folder.samples[idx]
        npy = feature_path(self.cache_root, image_path, self._image_root)
        feature = np.load(str(npy), allow_pickle=False)
        if feature.shape != (256, 768) or feature.dtype != np.float16:
            raise ValueError(f"Unexpected cache shape/dtype at {npy}")
        return image, label, torch.from_numpy(feature.copy())


# ---------------------------------------------------------------------------
# validation & cached_dataset
# ---------------------------------------------------------------------------

def cached_dataset(args, model):
    if not args.grounding_cache:
        raise ValueError("cached_recon requires --grounding_cache")
    root = Path(args.grounding_cache)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Cache manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    spec = manifest["spec"]

    if spec.get("version") != 3:
        raise ValueError(
            f"This directory uses cache version {spec.get('version')}, "
            "but this code expects version 3 (per-file). "
            "Re-export with the perfile variant of export_recon.py."
        )

    folder = ImageFolder(str(Path(args.data_path) / "train"))
    cache_cfg = getattr(args, "cache_cfg_dino", None)
    if cache_cfg is None:
        cache_cfg = args.cfg_dino
    expected = dict(
        version=3,
        transform="adm_center_noflip",
        layout="per_file",
        split="train",
        size=args.img_size,
        latent=args.latent_spec,
        count=len(folder),
        dataset=dataset_hash(folder),
        teacher=teacher_hash(model),
        weights=args.dino_weights,
        noise_scale=args.noise_scale_dino,
        t_eps=args.t_eps,
        steps=args.steps_dino,
        method=args.sampling_method_dino,
        cfg=cache_cfg,
        interval=[args.interval_min_dino, args.interval_max_dino],
    )
    for key, value in expected.items():
        if spec.get(key) != value:
            raise ValueError(f"Grounding cache mismatch for {key!r}: "
                             f"expected {value!r}, got {spec.get(key)!r}")

    # manifest id covers spec + count
    cache_id = digest([spec, manifest["count"]])
    if manifest.get("id") != cache_id:
        raise ValueError("Cache manifest id mismatch; re-export the cache.")
    if getattr(args, "expected_cache_id", cache_id) != cache_id:
        raise ValueError("Resumed cache_id does not match manifest.")
    if manifest["count"] != len(folder):
        raise ValueError(
            f"Cache count {manifest['count']} != dataset size {len(folder)}")

    args.cache_id = cache_id
    return CachedFolder(folder, root, spec)
