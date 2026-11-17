"""修复后的 CFG 检查：无条件分支必须不带语义，主分支必须带语义。"""
import os, sys, numpy as np, torch
from types import SimpleNamespace
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT); os.chdir(PROJECT_ROOT)
from main_jit import get_args_parser
from denoiser import Denoiser

IMAGENET = "/mnt/shared-storage-user/datafactory-share/gushishen/imagenet"
ARGV = ["--model","JiT-B/16","--img_size","256","--class_num","1000","--P_mean","-0.8",
        "--P_std","0.8","--t_eps","5e-2","--label_drop_prob","0.1","--noise_scale","1.0",
        "--sampling_method","euler","--num_sampling_steps","2","--cfg","2.9",
        "--interval_min","0.1","--interval_max","1.0","--output_dir","/tmp/_vcfg",
        "--data_path",IMAGENET,"--use_grounding","--grounding_dim","768","--grounding_source","val"]

device = torch.device("cuda:0"); torch.manual_seed(0)
args = get_args_parser().parse_args(ARGV)
model = Denoiser(args).to(device)
for n, p in model.named_parameters():
    if "adaLN_modulation" in n or "final_layer.linear" in n:
        torch.nn.init.normal_(p, std=0.02)
torch.nn.init.normal_(model.net.grounding_proj.weight, std=0.02)
torch.nn.init.normal_(model.net.grounding_proj.bias, std=0.02)
model.eval(); net = model.net

z = torch.randn(8,3,256,256, device=device)
t = torch.rand(8, device=device)*0.8+0.1
labels = torch.randint(0,1000,(8,), device=device)
p = model._provider(IMAGENET)
gt = model.encode_grounding(p.load_batch(np.arange(8)%1000, np.arange(8)%50).to(device))
null = torch.full_like(labels, model.num_classes)

print("="*70); print("1. cond / uncond 的条件构造"); print("="*70)
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    x_cond   = net(z,t,labels,grounding_tokens=gt).float()
    x_uncond = net(z,t,null, grounding_tokens=gt, grounding_mask=False).float()
    x_null   = net(z,t,null).float()                      # 纯 t+y
print("  uncond(gt + mask=False) 与 纯 t+y 的差: %.3e (应为 0)" % (x_uncond-x_null).abs().max())
print("  cond 与 uncond 的差:                     %.4f (应显著非零)"
      % (x_cond-x_uncond).abs().max())

print(); print("="*70); print("2. 语义项在条件里的能量占比"); print("="*70)
with torch.no_grad():
    c = net.t_embedder(t)+net.y_embedder(labels)
    proj = net.grounding_proj(gt)
    pc = c.unsqueeze(1)+proj
    d_cu = (c - (net.t_embedder(t)+net.y_embedder(null))).norm(dim=-1).mean()
print("  ||patch_condition|| = %.3f" % pc.norm(dim=-1).mean())
print("  ||c_cond - c_uncond|| = %.3f" % d_cu)
print("  相对差异 = %.2f%%  (修复前约 6.5%%)" % (100*d_cu/pc.norm(dim=-1).mean()))
print("  注：这个比例取决于训练后 W_g 的尺度；CFG 分支现在语义已被正确剥离。")

print(); print("="*70); print("3. DINO 编码耗时 (bsz 8)"); print("="*70)
import time
for tag, amp in (("fp32", False), ("bf16", True)):
    for _ in range(3):
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            model.grounding_encoder(z)
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(5):
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            model.grounding_encoder(z)
    torch.cuda.synchronize()
    print("  DINO %-4s %.1f ms" % (tag, (time.time()-t0)/5*1e3))
