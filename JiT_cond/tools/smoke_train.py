"""真实训练链路冒烟：数据加载 + DINO + forward/backward + EMA，确认 bsz=128 不 OOM。"""
import os, sys, time, math, json
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT); os.chdir(PROJECT_ROOT)

import torch, torchvision.transforms as transforms, torchvision.datasets as datasets
import util.misc as misc
from main_jit import get_args_parser
from denoiser import Denoiser
from util.crop import center_crop_arr

IMAGENET = "/mnt/shared-storage-user/datafactory-share/gushishen/imagenet"
ITERS = int(os.environ.get("SMOKE_ITERS", "8"))
BSZ = int(os.environ.get("SMOKE_BSZ", "128"))

args = get_args_parser().parse_args([
    "--model","JiT-B/16","--img_size","256","--class_num","1000","--P_mean","-0.8",
    "--P_std","0.8","--t_eps","5e-2","--label_drop_prob","0.1","--noise_scale","1.0",
    "--batch_size",str(BSZ),"--blr","5e-5","--weight_decay","0.0",
    "--output_dir","/tmp/_smoke","--data_path",IMAGENET,
    "--use_grounding","--grounding_dim","768","--grounding_source","val",
])
misc.init_distributed_mode(args)
device = torch.device("cuda", args.gpu)
torch.manual_seed(0)

tfm = transforms.Compose([transforms.Lambda(lambda im: center_crop_arr(im, args.img_size)),
                          transforms.RandomHorizontalFlip(), transforms.PILToTensor()])
ds = datasets.ImageFolder(os.path.join(IMAGENET, "train"), transform=tfm)
loader = torch.utils.data.DataLoader(ds, batch_size=BSZ, shuffle=True,
                                     num_workers=8, pin_memory=True, drop_last=True)
print("dataset:", len(ds), "images | bsz:", BSZ, "| iters:", ITERS)

model = Denoiser(args).to(device)
params = [p for p in model.parameters() if p.requires_grad]
model.ema_params1 = [p.detach().clone() for p in params]
model.ema_params2 = [p.detach().clone() for p in params]
opt = torch.optim.AdamW(params, lr=2e-4, betas=(0.9, 0.95))
model.train()

it = iter(loader)
losses = []
torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
for i in range(ITERS):
    x, labels = next(it)
    x = x.to(device, non_blocking=True).to(torch.float32).div_(255).mul_(2).sub_(1)
    labels = labels.to(device, non_blocking=True)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        loss = model(x, labels)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    model.update_ema()
    losses.append(loss.item())
    if i == 0:
        print("  iter0 %.2fs (含 warmup/compile)" % (time.time()-t0))
torch.cuda.synchronize()
dt = (time.time()-t0)/(ITERS-1)

print("\n===== 结果 =====")
print("loss: %.4f -> %.4f  (finite=%s)" % (losses[0], losses[-1], all(math.isfinite(l) for l in losses)))
print("稳态 %.0f ms/iter  (bsz %d, 单卡)" % (dt*1e3, BSZ))
print("峰值显存 %.0f MiB / %.0f MiB" % (torch.cuda.max_memory_allocated()/2**20,
                                        torch.cuda.get_device_properties(0).total_memory/2**20))
print("EMA 参数数: %d  与可训练参数一致: %s" % (len(model.ema_params1), len(model.ema_params1)==len(params)))
print("grounding_proj 已更新: %s" % bool(model.net.grounding_proj.weight.abs().sum() > 0))
