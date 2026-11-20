"""干净的稳态计时：跳过首次 compile，只测稳定后的每步耗时。"""
import os, sys, time
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT); os.chdir(PROJECT_ROOT)
import torch, torchvision.transforms as transforms, torchvision.datasets as datasets
import util.misc as misc
from main_jit import get_args_parser
from denoiser import Denoiser
from util.crop import center_crop_arr
torch._dynamo.config.cache_size_limit = 128

IMAGENET = "/mnt/shared-storage-user/datafactory-share/gushishen/imagenet"
BSZ = int(os.environ.get("BSZ", "128"))
WARM = int(os.environ.get("WARM", "3"))
MEAS = int(os.environ.get("MEAS", "8"))

args = get_args_parser().parse_args([
    "--model","JiT-B/16","--img_size","256","--class_num","1000","--P_mean","-0.8",
    "--P_std","0.8","--t_eps","5e-2","--label_drop_prob","0.1","--noise_scale","1.0",
    "--batch_size",str(BSZ),"--output_dir","/tmp/_timing","--data_path",IMAGENET,
    "--use_grounding","--grounding_dim","768","--grounding_source","val"])
misc.init_distributed_mode(args)
dev = torch.device("cuda", getattr(args,"gpu",0) or 0); torch.manual_seed(0)

tfm = transforms.Compose([transforms.Lambda(lambda im: center_crop_arr(im,256)),
                          transforms.RandomHorizontalFlip(), transforms.PILToTensor()])
ds = datasets.ImageFolder(os.path.join(IMAGENET,"train"), transform=tfm)
dl = torch.utils.data.DataLoader(ds, batch_size=BSZ, shuffle=True, num_workers=12,
                                 pin_memory=True, drop_last=True, persistent_workers=True)
it = iter(dl)

model = Denoiser(args).to(dev)
params = [p for p in model.parameters() if p.requires_grad]
model.ema_params1 = [p.detach().clone() for p in params]
model.ema_params2 = [p.detach().clone() for p in params]
opt = torch.optim.AdamW(params, lr=2e-4, betas=(0.9,0.95))
model.train()

def step():
    x, lb = next(it)
    x = x.to(dev, non_blocking=True).to(torch.float32).div_(255).mul_(2).sub_(1)
    lb = lb.to(dev, non_blocking=True)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        loss = model(x, lb)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    model.update_ema()
    return loss.item()

print("warmup %d 步（含 torch.compile）..." % WARM)
for i in range(WARM):
    t0 = time.time(); L = step(); torch.cuda.synchronize()
    print("  warmup %d: %.1fs  loss=%.4f" % (i, time.time()-t0, L))

torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize(); t0 = time.time()
losses = [step() for _ in range(MEAS)]
torch.cuda.synchronize()
dt = (time.time()-t0)/MEAS

print("\n===== 稳态 =====")
print("  %.0f ms/iter  (bsz=%d, 单卡)" % (dt*1e3, BSZ))
print("  loss %.4f -> %.4f" % (losses[0], losses[-1]))
print("  峰值显存 %.0f MiB / %.0f MiB" % (torch.cuda.max_memory_allocated()/2**20,
        torch.cuda.get_device_properties(0).total_memory/2**20))
n_img = len(ds)
print("\n===== 8 卡推算（全局 batch %d）=====" % (BSZ*8))
per_ep = n_img/(BSZ*8)
print("  每 epoch %.0f iter -> 约 %.1f 分钟" % (per_ep, per_ep*dt/60))
print("  80 epoch 约 %.1f 小时（不含 eval）" % (per_ep*dt*80/3600))
