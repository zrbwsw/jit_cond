"""GPU 端到端验证：val 配对采样、CFG 无分支一致性、训练 forward、DINO 路径。"""
import os
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from main_jit import get_args_parser
from denoiser import Denoiser

IMAGENET = "/mnt/shared-storage-user/datafactory-share/gushishen/imagenet"
BASE = [
    "--model", "JiT-B/16", "--img_size", "256", "--class_num", "1000",
    "--P_mean", "-0.8", "--P_std", "0.8", "--t_eps", "5e-2",
    "--label_drop_prob", "0.1", "--noise_scale", "1.0",
    "--sampling_method", "euler", "--num_sampling_steps", "2",
    "--cfg", "2.9", "--interval_min", "0.1", "--interval_max", "1.0",
    "--output_dir", "/tmp/_verify_fixes", "--data_path", IMAGENET,
    "--use_grounding", "--grounding_dim", "768",
]

device = torch.device("cuda:0")
torch.manual_seed(0)


def build(source):
    args = get_args_parser().parse_args(BASE + ["--grounding_source", source])
    return args, Denoiser(args).to(device)


print("=" * 74)
print("1. 训练 forward：grounding_mask 与 label dropout 联动")
print("=" * 74)
args, model = build("val")
model.train()
x = torch.randn(8, 3, 256, 256, device=device)
labels = torch.randint(0, 1000, (8,), device=device)
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    loss = model(x, labels)
loss.backward()
print("  loss = %.5f (finite=%s)" % (loss.item(), bool(torch.isfinite(loss))))
print("  grounding_proj grad norm = %.3e" % model.net.grounding_proj.weight.grad.norm().item())

# 强制 label_drop=1，验证 token 被关闭后梯度仍然有限
model.label_drop_prob = 1.0
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    loss_all_dropped = model(x, labels)
print("  label_drop=1.0 时 loss = %.5f (finite=%s)"
      % (loss_all_dropped.item(), bool(torch.isfinite(loss_all_dropped))))
model.label_drop_prob = 0.1

print()
print("=" * 74)
print("2. val 配对采样：token 来自同类验证集真实图片")
print("=" * 74)
model.eval()
labels_gen = torch.tensor([0, 0, 0, 999, 999], device=device)
intra = np.array([0, 1, 2, 0, 7])
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    t0 = time.time()
    out = model.generate(labels_gen, intra_indices=intra, data_path=IMAGENET)
    torch.cuda.synchronize()
    print("  生成 %s  用时 %.2fs  range [%.2f, %.2f]"
          % (tuple(out.shape), time.time() - t0, out.min().item(), out.max().item()))
    print("  非有限元素:", int((~torch.isfinite(out)).sum()))

# 确认不同参考图确实给出不同条件
from val_grounding import ValGroundingProvider
p = model._provider(IMAGENET)
imgs = p.load_batch(np.array([0, 0]), np.array([0, 1])).to(device)
tok = model.encode_grounding(imgs)
print("  同类两张不同参考图的 token 差异: L2 = %.3f (应显著非零)" % (tok[0] - tok[1]).norm())
imgs_ref = p.load_batch(np.array([0, 999]), np.array([0, 0])).to(device)
tok_ref = model.encode_grounding(imgs_ref)
print("  跨类参考图 token 差异:        L2 = %.3f" % (tok_ref[0] - tok_ref[1]).norm())

print()
print("=" * 74)
print("3. CFG 无条件分支：语义条件必须被关闭")
print("=" * 74)
z = torch.randn(4, 3, 256, 256, device=device)
t = torch.full((4, 1, 1, 1), 0.5, device=device)
gt = tok[:2].repeat(2, 1)
net = model.net
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    with_gt = net(z, t.flatten(), torch.full((4,), 1000, device=device),
                  grounding_tokens=gt, grounding_mask=False)
    without = net(z, t.flatten(), torch.full((4,), 1000, device=device), grounding_tokens=None)
d = (with_gt.float() - without.float()).abs().max().item()
print("  uncond(gt+mask=False) vs uncond(None): max|diff| = %.3e" % d)
print("  => 无条件分支不携带语义 token:", d < 1e-4)

with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    cond_out = net(z, t.flatten(), torch.full((4,), 0, device=device), grounding_tokens=gt)
dc = (cond_out.float() - without.float()).abs().max().item()
print("  cond(带语义) vs uncond(无语义): max|diff| = %.3e (应显著非零)" % dc)

print()
print("=" * 74)
print("4. 三种采样源都能跑通")
print("=" * 74)
for src in ["val", "probe", "none"]:
    model.grounding_source = src
    kw = dict(intra_indices=intra, data_path=IMAGENET) if src == "val" else {}
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        t0 = time.time()
        o = model.generate(labels_gen, **kw)
        torch.cuda.synchronize()
    print("  %-6s %s  %.2fs  finite=%s"
          % (src, tuple(o.shape), time.time() - t0, bool(torch.isfinite(o).all())))

print()
print("=" * 74)
print("5. val 模式下缺 intra_indices 应报错")
print("=" * 74)
model.grounding_source = "val"
try:
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        model.generate(labels_gen)
    print("  NO ERROR  <-- unexpected")
except ValueError as e:
    print("  ValueError:", e)
