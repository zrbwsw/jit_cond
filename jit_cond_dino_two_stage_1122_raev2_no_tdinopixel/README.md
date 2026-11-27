# Pixel JiT with RAEv2 DINOv3-L k=7 (_no_tdinopixel)

此目录使用 RAEv2 DINOv3-L k=7 encoder 提供 `[1024,16,16]` 条件 latent，训练 Pixel JiT 生成 RGB。默认训练条件来自对应真实图片的 clean latent；也可选择按 `t_dino_pixel ~ U[0.75,1)` 加噪。在线评估和独立采样使用 RAEv2 Stage 2 EMA 模型生成 latent，不需要训练 DinoDiT。

三种模式的 Pixel 条件均为 `E_pixel(t_pixel) + E_y(y) + W_g(z_cond)`。加噪时间只用于构造 latent，以及 pred 模式中冻结 DDT 的一步预测。

```bash
bash sh/train_pixel_raev2.sh --dry-run
GPU_HOST=agent-gpu-l2-1 bash sh/train_pixel_raev2.sh

# 可选：训练时使用 t_dino_pixel ~ U[0.75,1) 的加噪 grounding
CONDITION_MODE=noised NOISE_BETA_DINO_PIXEL=0.25 bash sh/train_pixel_raev2.sh

# 可选：加噪后先经过冻结的 RAEv2 DDT-XL 单步 clean prediction
CONDITION_MODE=pred NOISE_BETA_DINO_PIXEL=0.25 bash sh/train_pixel_raev2.sh
```

使用 noised/pred checkpoint 采样或续训时，也需设置相同的 `CONDITION_MODE`；旧 clean checkpoint 在新代码下仍默认按 clean 模式恢复。`pred` 仅训练时额外调用 DDT 单步预测，clean/noised 训练不加载 DDT。

默认 GPU1 8 卡、每卡 batch 128、160 epoch、Pixel Heun-50 CFG 2.9，每 20 epoch 在线评估。完整方法、资源路径和续训说明见 [README_raev2.md](README_raev2.md) 和 [中文改动说明](中文改动说明.md)。
