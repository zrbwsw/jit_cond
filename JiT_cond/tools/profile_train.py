"""拆分训练每一步的耗时：数据加载 / DINO 编码 / JiT fwd+bwd。"""
import os, sys, time
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT); os.chdir(PROJECT_ROOT)
import torch, torchvision.transforms as transforms, torchvision.datasets as datasets
import util.misc as misc
from main_jit import get_args_parser
from denoiser import Denoiser
from util.crop import center_crop_arr

IMAGENET = "/mnt/shared-storage-user/datafactory-share/gushishen/imagenet"
BSZ = int(os.environ.get("BSZ","128")); N = 6
args = get_args_parser().parse_args([
    "--model","JiT-B/16","--img_size","256","--class_num","1000","--P_mean","-0.8",
    "--P_std","0.8","--t_eps","5e-2","--label_drop_prob","0.1","--noise_scale","1.0",
    "--batch_size",str(BSZ),"--output_dir","/tmp/_p","--data_path",IMAGENET,
    "--use_grounding","--grounding_dim","768","--grounding_source","val"])
misc.init_distributed_mode(args)
dev = torch.device("cuda", args.gpu); torch.manual_seed(0)

tfm = transforms.Compose([transforms.Lambda(lambda im: center_crop_arr(im, 256)),
                          transforms.RandomHorizontalFlip(), transforms.PILToTensor()])
ds = datasets.ImageFolder(os.path.join(IMAGENET,"train"), transform=tfm)

def timeit(fn, n=N, warm=2):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time()-t0)/n*1e3

# --- dataloader ---
for nw in (0, 8, 12, 16):
    dl = torch.utils.data.DataLoader(ds, batch_size=BSZ, shuffle=True, num_workers=nw,
                                     pin_memory=True, drop_last=True, persistent_workers=bool(nw))
    it = iter(dl); next(it)
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(4): next(it)
    torch.cuda.synchronize()
    print("dataloader num_workers=%2d : %6.0f ms/batch" % (nw, (time.time()-t0)/4*1e3))
    del dl, it

model = Denoiser(args).to(dev); model.train()
net = model.net; enc = model.grounding_encoder
x0 = torch.randn(BSZ,3,256,256, device=dev)

print()
print("DINO encode  fp32 : %6.1f ms" % timeit(lambda: enc(x0)))
def _bf16():
    with torch.amp.autocast("cuda", dtype=torch.bfloat16): enc(x0)
print("DINO encode  bf16 : %6.1f ms" % timeit(_bf16))

from torch.optim import AdamW
opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
labels = torch.randint(0,1000,(BSZ,), device=dev)
gt = enc(x0)
def full_with():
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        loss = model(x0, labels)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
print("完整一步(含 DINO): %6.1f ms" % timeit(full_with))

def jit_only():
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = net(x0, torch.rand(BSZ,device=dev)*0.8+0.1, labels, grounding_tokens=gt)
        loss = out.pow(2).mean()
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
print("JiT fwd+bwd only : %6.1f ms" % timeit(jit_only))

def jit_nogt():
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = net(x0, torch.rand(BSZ,device=dev)*0.8+0.1, labels, grounding_tokens=None)
        loss = out.pow(2).mean()
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
print("JiT fwd+bwd 无条件: %6.1f ms" % timeit(jit_nogt))
