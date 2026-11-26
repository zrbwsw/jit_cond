# jit_cond_dino_two_stage_1122_raev2

## 方法

### 训练：真实图片 clean grounding

冻结 RAEv2 DINOv3-L k=7 encoder，聚合第 `[11,13,15,17,19,21,23]` 层，计算七层均值并加上最后一层的空间均值，再用 `stats.pt` 做 z-score 归一化，输出 `[B,1024,16,16]`。Pixel JiT 使用这些 patch tokens 学习 RGB 流匹配损失。类别 dropout 同时屏蔽类别和 grounding。

### 在线评估和采样：RAEv2 EMA latent grounding

`raev2_latent_sampler.py` 直接复用 RAEv2 官方 `instantiate_from_config`、transport、sampler 和 guidance 实现，严格加载 Stage 2 checkpoint 中的 `ema` 权重。`Denoiser.generate` 先采样 RAEv2 latent，再执行 Pixel Heun-50。

在线评估固定使用：

`Pixel_DIffusion/RAEv2/configs/stage2/sampling/imagenet-dinov3l-k7.yaml`

其配置为：100 步、CFG 1.0、IG 1.78、CFG 区间 `[0,1]`、IG 区间 `[0.1,1]`、latent `[1024,16,16]`。不要用 DinoDiT 的 25 步配置替换它。

## 本机资源

- DINOv3-L 权重：`Pixel_DIffusion/pretrain_models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`（已存在，无需下载）。
- DINOv3 源码：`Pixel_DIffusion/Ours_BaseT_rmsnorm2_Residual_1016/SSL_code/dinov3`。
- Latent stats：`stat/dinov3l_k7_imagenet1k_stats.pt`，软链接到 `RAEv2/stage1/imagenet/dinov3l-k7/stats.pt`。
- RAEv2 生成模型：`RAEv2/stage2/imagenet/dinov3l-k7/checkpoint.pt`，使用其中的 `ema` 分支。
- RAEv2 源码和采样配置：`Pixel_DIffusion/RAEv2`。

## 训练与续训

```bash
bash sh/train_pixel_raev2.sh --dry-run
GPU_HOST=agent-gpu-l2-1 bash sh/train_pixel_raev2.sh
```

默认输出：

`checkpoints/stage1-raev2-dinov3l-k7-stage2-pixeljit-b-e160`

默认每卡 batch 128、global batch 1024、`blr=5e-5`、160 epoch。每 5 epoch 更新 `checkpoint-last.pth`，每 80 epoch 保存编号 checkpoint，每 20 epoch 在线生成 50K 图片计算 FID/IS。

续训：

```bash
RESUME=$PWD/checkpoints/stage1-raev2-dinov3l-k7-stage2-pixeljit-b-e160/checkpoint-last.pth \
  GPU_HOST=agent-gpu-l2-1 bash sh/train_pixel_raev2.sh
```

`RESUME` 只恢复 Pixel JiT、optimizer、EMA 和 epoch。RAEv2 teacher 仍由 `RAEV2_CHECKPOINT` 和 `RAEV2_CONFIG` 指定。

## 验证

- `test_encoder.py`：真实 DINOv3-L k=7 前向，检查 `[2,1024,16,16]`。
- `tests/test_independent_training.py`：clean grounding、dropout、CFG、EMA、checkpoint 和官方采样配置。
- `raev2_latent_sampler.py`：已验证实际 EMA checkpoint 采样输出及有限性。

当前没有该方法的训练结果，不预设 FID/IS。
