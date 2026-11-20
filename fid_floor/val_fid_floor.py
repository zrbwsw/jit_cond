"""计算 ImageNet 50k 验证集真实图片的 FID/IS 下限（CPU，不占用 GPU）。

预处理与评估路径和训练/在线评估完全一致：

* 每张验证图 ``PIL.convert("RGB")`` 后经 ``util.crop.center_crop_arr`` 裁成
  256×256 uint8，与训练数据和 ``val_grounding.ValGroundingProvider`` 相同；
* 调用与 ``engine_jit.evaluate`` 相同的 ``torch_fidelity.calculate_metrics``，
  参考统计量为 ``jit_in256_stats.npz``。在线评估先把生成图写成 PNG 再由
  torch-fidelity 读回；PNG 无损，这里直接输入 uint8 张量，脚本开头会抽样
  验证两条路径逐像素一致。

另外把同一批特征与 ADM ``VIRTUAL_imagenet256_labeled.npz`` 中的 ``mu/sigma``
比较，并直接计算两份参考统计量之间的 Fréchet 距离，用于判断
``jit_in256_stats`` 的来源。

用法：python -I val_fid_floor.py --jit_cond_dir <含 util/crop.py 的目录>
"""

import argparse
import json
import os
import sys
import tempfile
import time
from multiprocessing import Pool

import cv2
import numpy as np
import torch
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_dir", default="/data0/datasets/imagenet/val")
    parser.add_argument("--fid_stats_dir", default="/data0/zhangruobei/Work/Pixel_Diffusion/fid_stats")
    parser.add_argument("--jit_cond_dir", required=True)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--decode_procs", type=int, default=32)
    parser.add_argument("--torch_threads", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache_root", required=True)
    return parser.parse_args()


_IMG_SIZE = None


def _init_worker(jit_cond_dir, img_size):
    global _IMG_SIZE, center_crop_arr
    sys.path.insert(0, jit_cond_dir)
    from util.crop import center_crop_arr as fn  # noqa: E402
    center_crop_arr = fn
    _IMG_SIZE = img_size


def _load(path):
    with Image.open(path) as im:
        return np.asarray(center_crop_arr(im.convert("RGB"), _IMG_SIZE), dtype=np.uint8)


class UInt8Dataset(torch.utils.data.Dataset):
    """torch-fidelity 期望的输入：uint8 ``[3,H,W]``。"""

    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return torch.from_numpy(np.ascontiguousarray(self.images[i].transpose(2, 0, 1)))


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(args.torch_threads)
    import torch_fidelity
    from torch_fidelity.datasets import TransformPILtoRGBTensor
    from torch_fidelity.metric_fid import fid_statistics_to_metric

    # 与 ValGroundingProvider 相同的排序：类名排序，类内文件名排序。
    classes = sorted(d for d in os.listdir(args.val_dir) if os.path.isdir(os.path.join(args.val_dir, d)))
    paths = []
    for cls in classes:
        d = os.path.join(args.val_dir, cls)
        paths += [os.path.join(d, f) for f in sorted(os.listdir(d))
                  if f.lower().endswith((".jpeg", ".jpg", ".png", ".bmp", ".webp"))]
    print(f"classes={len(classes)} images={len(paths)}", flush=True)
    assert len(classes) == 1000 and len(paths) == 50000

    t0 = time.time()
    images = np.empty((len(paths), args.img_size, args.img_size, 3), dtype=np.uint8)
    with Pool(args.decode_procs, initializer=_init_worker, initargs=(args.jit_cond_dir, args.img_size)) as pool:
        for i, arr in enumerate(pool.imap(_load, paths, chunksize=64)):
            images[i] = arr
    print(f"decoded+center-cropped in {time.time() - t0:.1f}s", flush=True)

    # 验证 PNG 往返（engine_jit: RGB->BGR cv2.imwrite；torch-fidelity: PIL 读回 RGB）
    # 与直接输入 uint8 张量逐像素一致。
    to_tensor = TransformPILtoRGBTensor()
    ds = UInt8Dataset(images)
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(0, len(paths), 997):
            p = os.path.join(tmp, f"{i:05d}.png")
            cv2.imwrite(p, images[i][:, :, ::-1])
            assert torch.equal(to_tensor(Image.open(p).convert("RGB")), ds[i]), f"PNG round-trip mismatch at {i}"
    print("PNG round-trip check: identical", flush=True)

    jit_stats = os.path.join(args.fid_stats_dir, "jit_in256_stats.npz")
    adm = np.load(os.path.join(args.fid_stats_dir, "VIRTUAL_imagenet256_labeled.npz"), mmap_mode="r")

    cache_name = f"imagenet_val50k_center{args.img_size}"
    # torch 2.6+ 的 torch.load 默认 weights_only=True，读不了 torch-fidelity 缓存的
    # numpy 统计量；特征缓存是张量可正常复用，统计量每次从特征重算（几秒）。
    stat_cache = os.path.join(args.cache_root, f"{cache_name}-inception-v3-compat-stat-fid-2048.pt")
    if os.path.exists(stat_cache):
        os.remove(stat_cache)
    t0 = time.time()
    m_jit = torch_fidelity.calculate_metrics(
        input1=ds, input2=None, fid_statistics_file=jit_stats, cuda=False, isc=True, fid=True,
        kid=False, prc=False, verbose=True, batch_size=args.batch_size, cache=True,
        cache_root=args.cache_root, input1_cache_name=cache_name)
    t_extract = time.time() - t0
    print("vs jit_in256_stats:", m_jit, flush=True)

    # 本次运行刚写出的统计缓存，可信。torch-fidelity 要求两侧 dtype 一致：
    # val/jit 统计为 float32，ADM 为 float64，统一转 float64。
    val_stats = torch.load(stat_cache, map_location="cpu", weights_only=False)
    f64 = lambda d: {"mu": np.asarray(d["mu"], np.float64), "sigma": np.asarray(d["sigma"], np.float64)}
    fd = lambda a, b: fid_statistics_to_metric(f64(a), f64(b), False)["frechet_inception_distance"]
    j = np.load(jit_stats)
    fid_adm = fd(val_stats, adm)
    fd_ref = fd(j, adm)
    print("vs ADM mu/sigma:", fid_adm, " FD(jit_in256_stats, ADM mu/sigma):", fd_ref, flush=True)

    result = {
        "num_images": len(paths),
        "preprocess": f"PIL RGB -> util.crop.center_crop_arr({args.img_size}) -> uint8",
        "torch_fidelity": torch_fidelity.__version__,
        "device": "cpu",
        "feature_extraction_seconds": round(t_extract, 1),
        "vs_jit_in256_stats": {"fid": m_jit["frechet_inception_distance"],
                               "is_mean": m_jit["inception_score_mean"],
                               "is_std": m_jit["inception_score_std"]},
        "vs_adm_mu_sigma": {"fid": fid_adm},
        "fd_jit_stats_vs_adm_stats": fd_ref,
        "jit_stats_mu_maxabs_diff_vs_adm": float(np.abs(j["mu"] - np.asarray(adm["mu"])).max()),
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
