"""诊断：LF 采样 latent 作为 grounding 为什么生成很差。

在同一批 (类别, 验证图) 上对比四种 grounding：
  dinov3_val : 训练时用的 DINOv3-B/16 x_norm_patchtokens（clean val 图）
  lf_encode  : LF 训练目标 = LF ckpt 内置 DINOv2-reg-B/14 编码同一张 val 图，
               去掉 LN affine、÷sqrt(var+eps)、×0.485（与 LatentForcing RAE.encode 一致）
  lf_sample  : LF 25 步采样出的 latent（与正在评测的 grounding 相同）
  none       : 不加 grounding（纯 t+y）
输出：逐元素/逐 token 统计、与 DINOv3 的相关性、grounding_proj 输出相对 t+y 的幅值，
以及 JiT ckpt-101 在每种 grounding 下的少量样本网格。

用法（单卡）：
  CUDA_VISIBLE_DEVICES=0 python tools/diagnose_lf_grounding.py --out <dir>
"""

import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from denoiser import Denoiser  # noqa: E402
from latent_force.sampler import LatentForcingSampler  # noqa: E402
from val_grounding import ValGroundingProvider  # noqa: E402

LF_CKPT = "/data0/zhangruobei/Work/LatentForcing/lfjit-l-200.pth"
JIT_CKPT = ("/data0/zhangruobei/Work/jit_cond_ablation/JiT_cond/checkpoints/"
            "JiT-B16-DINOv3B-PatchAdaLN-ValGrounding-B600/checkpoint-101.pth")
RAE_STAT = "/data0/zhangruobei/Work/Pixel_Diffusion/RAE/models/stats/dinov2/wReg_base/imagenet1k/stat.pt"


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--classes", default="1,88,207,279,388,417,812,980")
    p.add_argument("--per_class", type=int, default=4)
    p.add_argument("--gen_per_class", type=int, default=2)
    p.add_argument("--data_path", default="/data0/datasets/imagenet")
    p.add_argument("--dino_repo", default="/data0/zhangruobei/Work/Pixel_Diffusion/"
                   "Ours_BaseT_rmsnorm2_Residual_1016_normft_1028/SSL_code/dinov3")
    p.add_argument("--dino_weights", default="/data0/pretrain_models/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def build_lf_encoder(device):
    """从 LF ckpt 的 dino.* 重建其训练用 DINOv2-reg-B/14（无需联网）。"""
    from transformers import Dinov2WithRegistersConfig, Dinov2WithRegistersModel
    sd = torch.load(LF_CKPT, map_location="cpu", weights_only=False, mmap=True)["model_ema1"]
    pre = "dino.encoder.encoder."
    enc = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
    cfg = Dinov2WithRegistersConfig(image_size=518, patch_size=14, hidden_size=768,
                                    num_hidden_layers=12, num_attention_heads=12,
                                    num_register_tokens=4, layerscale_value=1.0)
    m = Dinov2WithRegistersModel(cfg)
    # LF 构造时把最后 LayerNorm 的 affine 去掉（normalize=True）
    m.layernorm = torch.nn.LayerNorm(768, eps=cfg.layer_norm_eps, elementwise_affine=False)
    missing, unexpected = m.load_state_dict(enc, strict=False)
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    stat = torch.load(RAE_STAT, map_location="cpu", weights_only=True)
    var = stat["var"].to(device)
    m = m.to(device).eval()

    @torch.no_grad()
    def encode(x01):  # x01: [B,3,256,256] in [0,1]
        x = F.interpolate(x01, size=(224, 224), mode="bicubic", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        z = m(pixel_values=(x - mean) / std).last_hidden_state[:, 5:]
        b, n, c = z.shape
        z = z.transpose(1, 2).reshape(b, c, 16, 16)
        z = z / torch.sqrt(var + 1e-5) * 0.485
        return z
    return encode


def stats(name, tok):  # tok: [B,256,768]
    t = tok.float()
    return {
        "name": name,
        "elem_std": float(t.std()),
        "elem_mean": float(t.mean()),
        "token_l2_mean": float(t.norm(dim=-1).mean()),
        "chan_mean_abs": float(t.mean(dim=(0, 1)).abs().mean()),
        "spatial_std_over_total": float((t - t.mean(dim=1, keepdim=True)).std() / t.std()),
    }


def main():
    a = parse()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    dev = torch.device("cuda")
    classes = [int(c) for c in a.classes.split(",")]
    labels = torch.tensor([c for c in classes for _ in range(a.per_class)], device=dev)
    intra = np.array([k for _ in classes for k in range(a.per_class)])

    prov = ValGroundingProvider(a.data_path, 256, 1000)
    imgs = prov.load_batch(labels.cpu().numpy(), intra).to(dev)  # [-1,1]

    # --- JiT ckpt-101 + DINOv3 编码器（与训练一致）
    args = SimpleNamespace(model="JiT-B/16", img_size=256, class_num=1000, attn_dropout=0.0, proj_dropout=0.0,
                           use_grounding=True, grounding_dim=768, grounding_model="dinov3_vitb16",
                           grounding_weights=a.dino_weights, grounding_repo=a.dino_repo,
                           grounding_source="val", ground_drop_prob=0.0, label_drop_prob=0.1,
                           P_mean=-0.8, P_std=0.8, t_eps=0.05, noise_scale=1.0,
                           ema_decay1=0.9999, ema_decay2=0.9996, sampling_method="heun",
                           num_sampling_steps=50, cfg=2.9, interval_min=0.1, interval_max=1.0)
    model = Denoiser(args)
    ck = torch.load(JIT_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    sd = model.state_dict()
    for n in names:  # 与 evaluate 相同：切到 EMA1
        sd[n] = ck["model_ema1"][n]
    model.load_state_dict(sd)
    model = model.to(dev).eval()

    with torch.no_grad():
        g_v3 = model.encode_grounding(imgs).float()                               # [B,256,768]
        lf_enc = build_lf_encoder(dev)
        g_lfenc = lf_enc(imgs * 0.5 + 0.5).flatten(2).transpose(1, 2).float()
        sampler = LatentForcingSampler(ckpt=LF_CKPT, latent_steps=25, sampling_method="heun", device=dev)
        g_lfs = sampler.sample_latent(labels).flatten(2).transpose(1, 2).float()

        rows = [stats("dinov3_val", g_v3), stats("lf_encode_val", g_lfenc), stats("lf_sample", g_lfs)]

        # LF 采样 vs LF 编码真实图：同类 latent 的分布是否接近（逐通道均值/方差）
        def chan(t):
            return t.mean(dim=(0, 1)), t.std(dim=(0, 1))
        m1, s1 = chan(g_lfenc)
        m2, s2 = chan(g_lfs)
        lf_consistency = {
            "chan_mean_corr": float(torch.corrcoef(torch.stack([m1, m2]))[0, 1]),
            "chan_std_ratio_median": float((s2 / s1).median()),
        }

        # 同一张图 DINOv3 与 LF 编码的逐 token 余弦（无对应关系时应接近 0）
        cos_v3_lfenc = float(F.cosine_similarity(g_v3, g_lfenc, dim=-1).mean())

        # grounding_proj 输出相对 t+y 的幅值（t=0.5，y=真实类别）
        net = model.net
        t = torch.full((labels.shape[0],), 0.5, device=dev)
        c = (net.t_embedder(t) + net.y_embedder(labels)).float()
        proj = {}
        for name, g in [("dinov3_val", g_v3), ("lf_encode_val", g_lfenc), ("lf_sample", g_lfs)]:
            s = net.grounding_proj(g.to(net.grounding_proj.weight.dtype)).float()
            proj[name] = {"proj_token_l2": float(s.norm(dim=-1).mean()),
                          "c_l2": float(c.norm(dim=-1).mean()),
                          "ratio": float(s.norm(dim=-1).mean() / c.norm(dim=-1).mean())}

    # --- 小样本生成网格（同噪声种子），比较四种 grounding
    import torchvision
    gl = torch.tensor([c for c in classes for _ in range(a.gen_per_class)], device=dev)
    gi = np.array([k for _ in classes for k in range(a.gen_per_class)])
    sel = torch.tensor([i for i in range(len(labels)) if intra[i] < a.gen_per_class], device=dev)
    grids = {}
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for name, g in [("dinov3_val", g_v3[sel]), ("lf_encode_val", g_lfenc[sel]),
                        ("lf_sample", g_lfs[sel]), ("none", None)]:
            torch.manual_seed(a.seed + 1)
            if g is None:
                out = model.generate(gl, grounding_source="none")
            else:
                out = model.generate(gl, grounding_tokens=g)
            out = ((out.float() + 1) / 2).clamp(0, 1)
            path = os.path.join(a.out, f"gen_{name}.png")
            torchvision.utils.save_image(out, path, nrow=a.gen_per_class * 4)
            grids[name] = path

    result = {"token_stats": rows, "lf_sample_vs_lf_encode": lf_consistency,
              "cos_dinov3_vs_lf_encode_same_image": cos_v3_lfenc,
              "grounding_proj_vs_ty": proj, "grids": grids, "classes": classes}
    with open(os.path.join(a.out, "diagnose.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
