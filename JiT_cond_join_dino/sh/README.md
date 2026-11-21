# JiT_cond_join_dino 启动脚本

`dinov2b` 和 `dinov3b` 各有独立训练、采样脚本。`_local` 在当前可见 GPU 上直接运行；`_h200` 通过现有 `run_on_gpu.sh` 转发到 GPU1（`agent-gpu-l2-1`），不需要修改连接配置。

## 统一默认配置

- JiT-B/16、256px、8 卡、每卡 batch 128（全局 1024）
- **200 epoch**
- `save_last_freq=5`：`checkpoint-last.pth` 每 5 个已完成 epoch 保存
- `save_interval=20`：持久 checkpoint 每 20 个已完成 epoch 保存
- 在线评估索引：`19、39、59、79、99`
- 输出目录：`checkpoints/JiT-B16-JoinDino-<编码器名>-B200/`
- `P_mean/std=-0.8/0.8`、`D_mean/std=-1.2/1.0`、`dino_weight=0.333`、`β=0.25`
- DINOv2-reg-B 从 `pretrain_models/dinov2-with-registers-base` 加载，已设置 `HF_HUB_OFFLINE=1`

## 训练

```bash
# DINOv2-reg-B/14：本机直接运行
bash sh/train_jit_b16_join_dinov2_8gpu_local.sh --dry-run
bash sh/train_jit_b16_join_dinov2_8gpu_local.sh

# DINOv2-reg-B/14：GPU1（H200）
bash sh/train_jit_b16_join_dinov2_8gpu_h200.sh --dry-run
bash sh/train_jit_b16_join_dinov2_8gpu_h200.sh

# DINOv3-B/16：本机直接运行
bash sh/train_jit_b16_join_dinov3_8gpu_local.sh --dry-run
bash sh/train_jit_b16_join_dinov3_8gpu_local.sh

# DINOv3-B/16：GPU1（H200）
bash sh/train_jit_b16_join_dinov3_8gpu_h200.sh --dry-run
bash sh/train_jit_b16_join_dinov3_8gpu_h200.sh
```

`EPOCHS`、`SAVE_INTERVAL`、`SAVE_LAST_FREQ`、`RDZV_PORT` 均可通过环境变量覆盖。

## 采样

```bash
# DINOv2-reg-B/14：本机或 GPU1 分别使用 _local/_h200
CHECKPOINT=/abs/path/checkpoint-xxx.pth \
CKPT_CFG=2.9 DINO_CFG=1.0 DINO_STEPS=25 \
  bash sh/sample_jit_b16_join_dinov2_8gpu_local.sh --dry-run
CHECKPOINT=/abs/path/checkpoint-xxx.pth \
CKPT_CFG=2.9 DINO_CFG=1.0 DINO_STEPS=25 \
  bash sh/sample_jit_b16_join_dinov2_8gpu_h200.sh --dry-run

# DINOv3-B/16：本机或 GPU1 分别使用 _local/_h200
CHECKPOINT=/abs/path/checkpoint-xxx.pth \
CKPT_CFG=2.9 DINO_CFG=1.0 DINO_STEPS=25 \
  bash sh/sample_jit_b16_join_dinov3_8gpu_local.sh --dry-run
CHECKPOINT=/abs/path/checkpoint-xxx.pth \
CKPT_CFG=2.9 DINO_CFG=1.0 DINO_STEPS=25 \
  bash sh/sample_jit_b16_join_dinov3_8gpu_h200.sh --dry-run

# DINOv3-B/16：DINO CFG 1.5，JiT CFG 2.9，依次评估 20/40/60/80/100 epoch
bash sh/sample_jit_b16_join_dinov3_dinocfg1p5_e20-100_8gpu_h200.sh --dry-run
bash sh/sample_jit_b16_join_dinov3_dinocfg1p5_e20-100_8gpu_h200.sh

# DINO CFG 2.0：等待上述 CFG 1.5 队列成功完成后执行同样的五个 checkpoint
bash sh/sample_jit_b16_join_dinov3_dinocfg2p0_e20-100_8gpu_h200.sh --dry-run
bash sh/sample_jit_b16_join_dinov3_dinocfg2p0_e20-100_8gpu_h200.sh

# DINO CFG 2.5：等待上述 CFG 2.0 队列成功完成后执行同样的五个 checkpoint
bash sh/sample_jit_b16_join_dinov3_dinocfg2p5_e20-100_8gpu_h200.sh --dry-run
bash sh/sample_jit_b16_join_dinov3_dinocfg2p5_e20-100_8gpu_h200.sh
```

每个样本先用 dino DiT 采 `DINO_STEPS` 步生成 grounding，再跑 50 步 JiT。`CKPT_CFG` 是 JiT path 的 CFG，`DINO_CFG` 是 dino path 的 CFG（1.0 = 不加引导）。

`--dry-run` 只打印最终命令，不启动训练、采样或 SSH。Python、数据目录、DINOv3 权重等路径已在脚本顶部配置为当前共享资源。
