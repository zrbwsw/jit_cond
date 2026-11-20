"""直接跑一次真实 evaluate()，确认真实语义 token 的采样链路端到端可用。"""
import os, sys, torch, numpy as np
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT); os.chdir(PROJECT_ROOT)

import util.misc as misc
from main_jit import get_args_parser
from denoiser import Denoiser
from engine_jit import evaluate

IMAGENET = "/mnt/shared-storage-user/datafactory-share/gushishen/imagenet"
argv = ["--model","JiT-B/16","--img_size","256","--class_num","1000","--P_mean","-0.8",
        "--P_std","0.8","--t_eps","5e-2","--label_drop_prob","0.1","--noise_scale","1.0",
        "--sampling_method","euler","--num_sampling_steps","2","--cfg","2.9",
        "--interval_min","0.1","--interval_max","1.0",
        "--output_dir","/tmp/_verify_eval","--data_path",IMAGENET,
        "--use_grounding","--grounding_dim","768","--grounding_source","val",
        "--num_images","1000","--gen_bsz","50"]

torch.manual_seed(0)
args = get_args_parser().parse_args(argv)
# 交给 torch.distributed.run 初始化单进程组，否则 evaluate() 里的 barrier 会报错。
misc.init_distributed_mode(args)

device = torch.device("cuda:0")
model = Denoiser(args).to(device)
model.ema_params1 = [p.detach().clone() for p in model.parameters() if p.requires_grad]
model.eval()

print("grounding_source =", model.grounding_source)
print("num_images=%d gen_bsz=%d steps=%d" % (args.num_images, args.gen_bsz, model.steps))
os.makedirs(args.output_dir, exist_ok=True)

# log_writer=None 跳过 FID 计算，只验证生成链路
evaluate(model, args, epoch=0, batch_size=args.gen_bsz, log_writer=None)

d = os.path.join(args.output_dir, [
    x for x in os.listdir(args.output_dir) if x.startswith("euler-steps")
][0])
files = sorted(os.listdir(d))
print("\n生成图片数:", len(files))
print("示例:", files[:3])
from PIL import Image
im = np.array(Image.open(os.path.join(d, files[0])))
print("首张图 shape=%s dtype=%s 像素范围=[%d, %d] 非零比例=%.1f%%"
      % (im.shape, im.dtype, im.min(), im.max(), 100*(im.sum(axis=2)>0).mean()))
# 检查不是全同一个画面
im2 = np.array(Image.open(os.path.join(d, files[500])))
print("第 500 张与首张差异 mean|d| = %.2f" % np.abs(im.astype(int)-im2.astype(int)).mean())
