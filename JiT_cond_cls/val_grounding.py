"""验证集真实图片的 clean semantic token 提供器。

上限实验的设定是：采样时也始终能拿到干净语义。这里把生成队列和验证集
做一对一配对——第 g 张待生成图片属于类别 ``g // per_class``，取该类别
验证集的第 ``g % per_class`` 张真实图片作为语义来源。

配对方式与 ``engine_jit.evaluate`` 的 ``class_label_gen_world`` 完全一致，
因此每个生成样本都对应一张同类别的真实参考图，类别永不串位。
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
import torch
from PIL import Image

from util.crop import center_crop_arr


class ValGroundingProvider:
    """按类别与类内序号给出验证集真实图片（[-1,1]，与训练同分布）。"""

    def __init__(self, root: str, img_size: int, num_classes: int):
        self.root = root
        self.img_size = img_size
        self.num_classes = num_classes
        val_dir = os.path.join(root, "val")
        if not os.path.isdir(val_dir):
            raise FileNotFoundError(f"validation split not found: {val_dir}")

        # ImageFolder 按类名排序建立 label；这里用同样的顺序，保证
        # label c 对应的就是 ImageFolder 的第 c 个类。
        self.classes = sorted(
            d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))
        )
        if len(self.classes) != num_classes:
            raise ValueError(
                f"val split has {len(self.classes)} classes, expected {num_classes}"
            )
        self._per_class: List[List[str]] = []
        for cls in self.classes:
            d = os.path.join(val_dir, cls)
            files = sorted(
                f for f in os.listdir(d)
                if f.lower().endswith((".jpeg", ".jpg", ".png", ".bmp", ".webp"))
            )
            if not files:
                raise FileNotFoundError(f"no images in {d}")
            self._per_class.append(files)

    def path_for(self, label: int, intra_index: int) -> str:
        files = self._per_class[label]
        # 类内图片数不足时循环复用，避免越界；50000/1000=50 正好用满。
        return os.path.join(
            os.path.join(self.root, "val", self.classes[label]), files[int(intra_index) % len(files)]
        )

    def load_batch(self, labels: np.ndarray, intra_indices: np.ndarray) -> torch.Tensor:
        """返回 [B,3,H,W]、取值范围 [-1,1] 的真实图片。"""
        imgs = []
        for label, k in zip(labels, intra_indices):
            with Image.open(self.path_for(int(label), int(k))) as im:
                arr = np.array(center_crop_arr(im.convert("RGB"), self.img_size))
            imgs.append(torch.from_numpy(arr).permute(2, 0, 1))
        batch = torch.stack(imgs).float().div(255.0).mul(2.0).sub(1.0)
        return batch
