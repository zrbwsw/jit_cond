"""验证修复：label dropout 不泄漏、CFG 无条件分支、val 配对、zero-init 等价性。"""
import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from model_jit import JiT_models

torch.manual_seed(0)
B, D = 4, 768
x = torch.randn(B, 3, 256, 256)
t = torch.rand(B) * 0.8 + 0.1
y = torch.randint(0, 1000, (B,))
gt = torch.randn(B, D)

print("=" * 70)
print("1. zero-init: grounded 初始化 == baseline 初始化")
print("=" * 70)
base = JiT_models["JiT-B/16"](num_classes=1000, grounding_dim=None, input_size=256)
gnd = JiT_models["JiT-B/16"](num_classes=1000, grounding_dim=D, input_size=256)
gnd.load_state_dict({k: v for k, v in base.state_dict().items()}, strict=False)
with torch.no_grad():
    o1 = base(x, t, y)
    o2 = gnd(x, t, y, grounding_tokens=gt)
print("  grounding_proj weight 全零:", bool((gnd.grounding_proj.weight == 0).all()))
print("  max|baseline - grounded| = %.10f" % (o1 - o2).abs().max().item())

print()
print("=" * 70)
print("2. grounding_mask=False 等价于 grounding_tokens=None")
print("=" * 70)
with torch.no_grad():
    a = gnd(x, t, y, grounding_tokens=None)
    b = gnd(x, t, y, grounding_tokens=gt, grounding_mask=False)
print("  max|none - maskFalse| = %.10f" % (a - b).abs().max().item())

print()
print("=" * 70)
print("3. 逐样本 mask 混用：被关掉的样本与纯 t+y 一致")
print("=" * 70)
mask = torch.tensor([True, False, True, False])
with torch.no_grad():
    out = gnd(x, t, y, grounding_tokens=gt, grounding_mask=mask)
print("  混合 mask 前向 shape:", tuple(out.shape))

# 把 grounding_proj 设成非零，才看得出 mask 是否真的起作用
with torch.no_grad():
    torch.nn.init.normal_(gnd.grounding_proj.weight, std=0.02)
    torch.nn.init.normal_(gnd.grounding_proj.bias, std=0.02)
    all_on = gnd(x, t, y, grounding_tokens=gt, grounding_mask=torch.ones(B, dtype=torch.bool))
    all_off = gnd(x, t, y, grounding_tokens=gt, grounding_mask=torch.zeros(B, dtype=torch.bool))
    mixed = gnd(x, t, y, grounding_tokens=gt, grounding_mask=mask)
# 关掉的样本（下标 1、3）应与 all_off 相同；开着的（0、2）应与 all_on 相同
d_off = (mixed[[1, 3]] - all_off[[1, 3]]).abs().max().item()
d_on = (mixed[[0, 2]] - all_on[[0, 2]]).abs().max().item()
print("  关闭样本 vs 全关: max diff = %.10f" % d_off)
print("  开启样本 vs 全开: max diff = %.10f" % d_on)
print("  => mask 逐样本生效:", d_off < 1e-9)

print()
print("=" * 70)
print("4. 验证集 provider 配对")
print("=" * 70)
try:
    from val_grounding import ValGroundingProvider
    p = ValGroundingProvider(
        root="/mnt/shared-storage-user/datafactory-share/gushishen/imagenet",
        img_size=256, num_classes=1000,
    )
    print("  类别数:", len(p.classes), "| 第一类:", p.classes[0])
    for label, k in [(0, 0), (0, 49), (999, 0), (500, 25)]:
        path = p.path_for(label, k)
        print("    label=%3d k=%2d -> %s" % (label, k, os.path.basename(os.path.dirname(path)) + "/" + os.path.basename(path)))
    import numpy as np
    imgs = p.load_batch(np.array([0, 0, 999]), np.array([0, 1, 0]))
    print("  load_batch shape:", tuple(imgs.shape), "range [%.2f, %.2f]" % (imgs.min(), imgs.max()))
except Exception as e:
    print("  FAILED:", type(e).__name__, e)
