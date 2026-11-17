"""在"训练后"的非零初始化状态下验证语义条件真的起作用、且 CFG 分离干净。"""
import os, sys, numpy as np, torch, torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT); os.chdir(PROJECT_ROOT)
from main_jit import get_args_parser
from denoiser import Denoiser

IMAGENET = "/mnt/shared-storage-user/datafactory-share/gushishen/imagenet"
BASE = ["--model","JiT-B/16","--img_size","256","--class_num","1000","--P_mean","-0.8",
        "--P_std","0.8","--t_eps","5e-2","--label_drop_prob","0.1","--noise_scale","1.0",
        "--sampling_method","euler","--num_sampling_steps","2","--cfg","2.9",
        "--interval_min","0.1","--interval_max","1.0","--output_dir","/tmp/_vs",
        "--data_path",IMAGENET,"--use_grounding","--grounding_dim","768","--grounding_source","val"]

def dezero(m):
    """模拟训练后的状态：adaLN 与 final linear 不再是零。"""
    for name, p in m.named_parameters():
        if "adaLN_modulation" in name or name.startswith("net.final_layer.linear") or "final_layer.linear" in name:
            nn.init.normal_(p, std=0.02)
    nn.init.normal_(m.net.grounding_proj.weight, std=0.02)
    nn.init.normal_(m.net.grounding_proj.bias, std=0.02)

device = torch.device("cuda:0"); torch.manual_seed(0)
args = get_args_parser().parse_args(BASE)
model = Denoiser(args).to(device); dezero(model); model.eval()
net = model.net

print("="*72); print("1. 语义条件是否真的改变输出"); print("="*72)
z = torch.randn(4,3,256,256, device=device); t = torch.full((4,),0.5, device=device)
labels = torch.zeros(4, dtype=torch.long, device=device)
p = model._provider(IMAGENET)
gt = model.encode_grounding(p.load_batch(np.array([0,0,0,0]), np.array([0,1,2,3])).to(device))
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    o_gt   = net(z,t,labels,grounding_tokens=gt).float()
    o_none = net(z,t,labels,grounding_tokens=None).float()
    o_off  = net(z,t,labels,grounding_tokens=gt, grounding_mask=False).float()
print("  带语义 vs 无 grounding : max|d| = %.4f  (应显著非零)" % (o_gt-o_none).abs().max())
print("  mask=False vs None     : max|d| = %.3e (应恰好 0)" % (o_off-o_none).abs().max())

print(); print("="*72); print("2. 不同参考图 -> 不同输出（语义逐样本生效）"); print("="*72)
gt_a = model.encode_grounding(p.load_batch(np.array([0,0]), np.array([0,1])).to(device))
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    oa = net(z[:2],t[:2],labels[:2],grounding_tokens=gt_a).float()
print("  同 batch 内两张不同参考图 -> 输出差异 max|d| = %.4f" % (oa[0]-oa[1]).abs().max())

print(); print("="*72); print("3. 训练 forward 的梯度是否到达语义分支"); print("="*72)
model.train()
x = torch.randn(8,3,256,256, device=device); lb = torch.randint(0,1000,(8,), device=device)
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    loss = model(x, lb)
loss.backward()
for n in ["net.grounding_proj.weight","net.t_embedder.mlp.0.weight",
          "net.y_embedder.embedding_table.weight","net.blocks.0.adaLN_modulation.1.weight"]:
    pp = dict(model.named_parameters())[n]
    print("  %-42s |grad| = %.4e" % (n, 0.0 if pp.grad is None else pp.grad.norm().item()))

print(); print("="*72); print("4. label dropout 泄漏检查：强制全丢标签时语义是否被关闭"); print("="*72)
model.label_drop_prob = 1.0
captured = {}
orig = model.net.forward
def spy(xx, tt, yy, grounding_tokens=None, grounding_mask=None, **kw):
    captured["mask"] = grounding_mask
    return orig(xx, tt, yy, grounding_tokens=grounding_tokens, grounding_mask=grounding_mask, **kw)
model.net.forward = spy
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    model(x, lb)
model.net.forward = orig
m = captured["mask"]
print("  传入的 grounding_mask:", m)
print("  全为 False（语义被关闭）:", bool(m is not None and not m.any()))

print(); print("="*72); print("5. ground_drop_prob 生效"); print("="*72)
model.label_drop_prob = 0.0; model.ground_drop_prob = 1.0
model.net.forward = spy; captured.clear()
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    model(x, lb)
model.net.forward = orig
m = captured["mask"]
print("  ground_drop_prob=1.0 -> mask 全 False:", bool(m is not None and not m.any()))
model.ground_drop_prob = 0.0
model.net.forward = spy; captured.clear()
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    model(x, lb)
model.net.forward = orig
m = captured["mask"]
print("  ground_drop_prob=0.0 -> mask 全 True :", bool(m is not None and m.all()))
