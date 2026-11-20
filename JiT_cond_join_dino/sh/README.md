# JiT_cond_join_dino 启动脚本

本机 8 卡直接用 `torchrun`（c10d rendezvous，`--rdzv-endpoint 127.0.0.1:<port>`
避免与并行任务抢端口）。DINOv2-reg-B 从 HF 本地缓存加载，脚本内已设
`HF_HUB_OFFLINE=1`。

## 训练

```bash
bash sh/train_jit_b16_join_dino_8gpu_local.sh --dry-run
bash sh/train_jit_b16_join_dino_8gpu_local.sh
```

默认 JiT-B/16、256px、8 卡、每卡 batch 128（全局 1024）、600 epoch。两条路径：
JiT path 的 `t` 用 `P_mean/std=-0.8/0.8`；dino path 的 `t_dino` 用
`D_mean/std=-1.2/1.0`，`dino_weight=0.333`，cascaded_noised `β=0.25`。
输出和 `train.log` 默认在 `checkpoints/JiT-B16-JoinDino-DINOv2regB-B600/`。
可用 `DINO_CFG=` 覆盖 dino path 训练无关项（采样才用）。

## 采样

```bash
CHECKPOINT=/abs/path/checkpoint-xxx.pth \
CKPT_CFG=2.9 DINO_CFG=1.0 DINO_STEPS=25 \
  bash sh/sample_jit_b16_join_dino_8gpu_local.sh --dry-run
bash sh/sample_jit_b16_join_dino_8gpu_local.sh
```

每个样本先用 dino DiT 采 `DINO_STEPS` 步生成 grounding（instance-level），再跑
50 步 JiT。`CKPT_CFG` 是 JiT path 的 CFG，`DINO_CFG` 是 dino path 的 CFG
（1.0 = 不加引导）。评测目录名里带上了这两个 CFG 和 dino 步数。

`--dry-run` 只打印最终命令。GPU 卡号、Python、数据目录等都在脚本顶部修改。
