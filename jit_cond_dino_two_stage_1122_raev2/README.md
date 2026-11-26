# Pixel JiT with RAEv2 DINOv3-L k=7

此目录使用 RAEv2 DINOv3-L k=7 encoder 提供 `[1024,16,16]` 条件 latent，训练 Pixel JiT 生成 RGB。训练条件来自对应真实图片的 clean latent；在线评估和独立采样使用 RAEv2 Stage 2 EMA 模型生成 latent，不需要训练 DinoDiT。

```bash
bash sh/train_pixel_raev2.sh --dry-run
GPU_HOST=agent-gpu-l2-1 bash sh/train_pixel_raev2.sh
```

默认 GPU1 8 卡、每卡 batch 128、160 epoch、Pixel Heun-50 CFG 2.9，每 20 epoch 在线评估。完整方法、资源路径和续训说明见 [README_raev2.md](README_raev2.md) 和 [中文改动说明](中文改动说明.md)。
