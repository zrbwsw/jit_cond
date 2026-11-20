# JiT-B + 独立 dino DiT 联合去噪（join-dino）

本目录从 `JiT_cond` 的 patch-level AdaLN 出发，把 grounding 的来源从"clean
image 过冻结 DINOv3"换成一个**独立训练的 dino DiT**：它去噪 DINOv2 latent，
采样时先自行从噪声生成一份 grounding，再交给 JiT。等价于把 Latent Forcing 的
dino path 拆成独立网络，并用 `c = t + y + grounding` 的方式注入 JiT。

```text
c[b, patch] = t_embed[b] + y_embed[b] + t_dino_embed[b] + W_g · grounding[b, patch]
```

## 为什么这样做

instance-level 的 clean grounding 已被证明对生成有益：同一份 JiT-B/16 权重
（ckpt-101），用每样本一张验证图的 DINOv3 grounding 采样得到 FID 3.58，优于
JiT baseline 600 epoch 的 3.66。但那依赖采样时有真实参考图。join-dino 用一个
生成式的 dino DiT 替代"真实参考图 → DINOv3"，让 grounding 在无参考图时也能
instance-level 地产生：

* 本质是**双倍网络参数**：JiT-B（≈132M）+ dino DiT（≈131M，JiT-B 同尺寸），
  冻结 DINOv2-reg-B（≈87M）只做训练目标编码，不计入可训练参数或 EMA。
* 采样步数变多（先 dino 25 步，再 JiT 50 步），这是预期代价。

## 两条路径

| 路径 | 输入 | 目标 | 条件 |
| --- | --- | --- | --- |
| JiT (pixel) | 带噪图像 `z` | 干净图像的 x-prediction | `t + y + t_dino + W_g·grounding` |
| dino DiT | 带噪 DINOv2 latent `z_dino` | 干净 DINOv2 latent 的 x-prediction | `t_dino + y` |

两个网络参数独立，不共享 trunk；每个样本两路都监督，
`loss = loss_jit + dino_weight · loss_dino`（`dino_weight=0.333`，follow LF）。

### DINOv2 latent 空间（与 Latent Forcing 完全一致）

`latent_force/dinov2_encoder.py` 复刻 LatentForcing 的 `RAE.encode`：

```text
clean 图 [0,1] → bicubic 224² → ImageNet mean/std
→ facebook/dinov2-with-registers-base（最后 LayerNorm 去掉 affine）
→ 丢 1 CLS + 4 register → [B,768,16,16]
→ (z − 0) / sqrt(var + 1e-5)      # RAE ImageNet-1k 统计，mean=None
→ × 0.485                         # match_pixel_norm
```

该 encoder 与 LF 预训练权重 `lfjit-l-200.pth` 内置的 `dino.*` 在同一张图上逐
元素一致（`max|diff|=0`）。dino DiT 预测的目标就是这个归一化、乘过 0.485 的
clean latent，与 LF 的预测目标相同。dino path 和 LF 一样经过 bottleneck
（`BottleneckPatchEmbed`，patch=1 的两层 1×1 卷积）。

### 训练时 JiT 见到的 grounding（LF cascaded_noised）

JiT 不是直接吃 clean latent，而是吃带轻噪声的版本：`t_dino ∈ [1−β, 1)`
（`β = dino_noise_beta = 0.25`），并把 `t_dino` 一并加进条件 `c`。这缩小了
训练（clean-ish latent）与采样（dino DiT 生成的 latent）之间的分布差距。

`t` 调度：JiT path 沿用 JiT_cond（`P_mean=-0.8, P_std=0.8`）；dino path 的
`t_dino` follow LF（`D_mean=-1.2, D_std=1.0`）。

## 采样（instance-level）

每个样本先用 dino DiT 从纯噪声跑 `--dino_steps`（默认 25）步，得到自己的
`[B,768,16,16]` grounding（终点 `t_dino=1`），flatten 成 `[B,256,768]` 后，
整条 JiT 轨迹（50 步）复用这一份。因为 grounding 逐样本生成，所以是
instance-level，不读验证集、不需要参考图。

### CFG

* **JiT path**：与 JiT_cond 一致。cond 分支带 y 和 grounding；uncond 分支用
  null 类，并关闭 grounding（`grounding_mask=False`，`t_dino` 也随之关闭），
  退回纯 `t + y`。
* **dino path**：保留独立 CFG。训练时 label 按 `label_drop_prob` 丢弃，采样
  用 `--dino_cfg`（默认 1.0 = 不加引导；`--dino_cfg 1.5` 等启用）。

## 本地资源

```text
ImageNet:     /data0/datasets/imagenet
FID stats:    /data0/zhangruobei/Work/Pixel_Diffusion/fid_stats
DINOv2-reg-B: facebook/dinov2-with-registers-base（HF 本地缓存，HF_HUB_OFFLINE=1）
RAE 统计:     latent_force/dinov2_wreg_base_imagenet1k_stat.pt（随仓库提供）
```

没有兼容的预训练 checkpoint，join-dino 从头训练。

## 8-GPU 实验

训练默认 JiT-B/16、256px、8 卡、每卡 batch 128（全局 1024）、600 epoch，
绝对学习率 `blr × global_batch / 256`，`blr=5e-5` → `2e-4`。

```bash
bash sh/train_jit_b16_join_dino_8gpu_local.sh --dry-run
bash sh/train_jit_b16_join_dino_8gpu_local.sh
```

采样前在脚本顶部（或用环境变量）设置 `CHECKPOINT`、`CKPT_CFG`、`DINO_CFG`、
`DINO_STEPS`：

```bash
bash sh/sample_jit_b16_join_dino_8gpu_local.sh --dry-run
bash sh/sample_jit_b16_join_dino_8gpu_local.sh
```

`--dry-run` 只打印最终命令。GPU 卡号、数据路径、dino path 配置都在脚本顶部修改。
