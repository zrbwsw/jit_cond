import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import functional as TF

from util.crop import center_crop_arr


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else stream_hash(stream)


def stream_hash(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(4 << 20), b""):
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


def image_view(folder, idx, flip, size):
    path, label = folder.samples[idx]
    image = center_crop_arr(folder.loader(path), size)
    if flip:
        image = TF.hflip(image)
    return TF.pil_to_tensor(image), label


def shard_path(root, number):
    return root / f"part-{number:06d}.npy"


@lru_cache(maxsize=8)
def open_shard(path):
    return np.load(path, mmap_mode="r", allow_pickle=False)


class CachedFolder(torch.utils.data.Dataset):
    def __init__(self, folder, root, spec):
        self.folder, self.root, self.spec = folder, root, spec

    def __len__(self):
        return len(self.folder)

    def __getitem__(self, idx):
        s = self.spec
        view = torch.randint(s["views"], ()).item() if s["views"] == 2 else 0
        image, label = image_view(self.folder, idx, view, s["size"])
        part, offset = divmod(idx, s["shard_size"])
        feature = open_shard(shard_path(self.root, part))[offset, view]
        return image, label, torch.from_numpy(feature.copy())


def cached_dataset(args, model):
    if not args.grounding_cache:
        raise ValueError("cached_recon requires --grounding_cache")
    root = Path(args.grounding_cache)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    spec = manifest["spec"]
    cache_id = digest([spec, manifest["shards"]])
    if manifest["id"] != cache_id or getattr(args, "expected_cache_id", cache_id) != cache_id:
        raise ValueError("Wrong cache manifest or resumed cache_id")
    folder = ImageFolder(str(Path(args.data_path) / "train"))
    expected = dict(version=1, split="train", size=args.img_size, latent=args.latent_spec,
                    count=len(folder), dataset=dataset_hash(folder), teacher=teacher_hash(model),
                    weights=args.dino_weights, noise_scale=args.noise_scale_dino, t_eps=args.t_eps,
                    steps=args.steps_dino, method=args.sampling_method_dino, cfg=args.cfg_dino,
                    interval=[args.interval_min_dino, args.interval_max_dino])
    for key, value in expected.items():
        if spec.get(key) != value:
            raise ValueError(f"Grounding cache mismatch for {key}")
    if spec["views"] not in (1, 2) or spec["shard_size"] < 1:
        raise ValueError("Invalid view or shard configuration")
    count = (len(folder) + spec["shard_size"] - 1) // spec["shard_size"]
    if len(manifest["shards"]) != count:
        raise ValueError("Incomplete cache manifest")
    for i, checksum in enumerate(manifest["shards"]):
        path = shard_path(root, i)
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        rows = min(spec["shard_size"], len(folder) - i * spec["shard_size"])
        if arr.shape != (rows, spec["views"], 256, 768) or arr.dtype != np.float16:
            raise ValueError(f"Invalid shard: {path}")
        if getattr(args, "verify_cache_hashes", False) and file_hash(path) != checksum:
            raise ValueError(f"Damaged shard: {path}")
    args.cache_id = cache_id
    return CachedFolder(folder, root, spec)
