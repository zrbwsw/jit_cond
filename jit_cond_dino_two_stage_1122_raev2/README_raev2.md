# jit_cond_dino_two_stage_1122_raev2

**使用 RAEv2 DINOv3-L k=7 encoder 作为 Pixel JiT 的条件**

---

## 核心设计

### Stage 1：DINOv3-L k=7 Encoder（冻结，无需训练）

- **只用 encoder**：DINOv3-L 聚合 7 层特征 [12,14,16,18,20,22,24]
- **输出**：[B, 1024, 16, 16] z-score 归一化的 latent
- **不需要 decoder**：Pixel JiT 本身就是 decoder（latent → pixel）

### Stage 2：Pixel JiT（需要训练）

**训练**：
```python
# 条件：使用 clean image 的 latent（保留完整实例信息）
clean_latent = dinov3_k7_encoder.encode(clean_image)  # [B, 1024, 16, 16]

# 像素去噪训练
z_pixel = t * clean_image + (1 - t) * noise
x_pred = pixel_jit(z_pixel, t, label, grounding=clean_latent)
loss = v_loss(x_pred, clean_image, z_pixel, t)
```

**推理**：
```python
# 步骤 1：采样语义 latent（调用 RAEv2 Stage 1 完整采样）
z_latent = raev2_stage1_sample(randn(), labels)  # 需要 RAEv2 环境

# 步骤 2：以 z_latent 为条件，采样像素
image = pixel_jit_sample(randn(), labels, grounding=z_latent)
```

**关键优势**：
- ✅ 训练用 clean latent，保留完整实例信息
- ✅ 推理用 RAEv2 采样的 latent，保证语义合理
- ✅ 两者都是归一化后的分布，训练/推理对齐
- ✅ 无需训练 Stage 1，直接复用 RAEv2 预训练

---

## 与原版 DinoDiT 的对比

| 项目 | 原版（DinoDiT） | RAEv2 版本 |
|------|----------------|-----------|
| **Stage 1** | 训练 DinoDiT（E160） | **无需训练**，直接用 DINOv3-L k=7 |
| **Encoder** | DINOv3-B/16 单层 | DINOv3-L/16 k=7（7层聚合） |
| **Latent 维度** | [B, 768, 16, 16] | [B, 1024, 16, 16] |
| **训练条件** | noised/pred/cached | **clean latent**（最强实例信息） |
| **推理条件** | DinoDiT 采样 | RAEv2 Stage 1 采样 |
| **Decoder** | DinoDiT decoder | **Pixel JiT 本身** |

---

## 使用方法

### 1. 下载统计文件（必需）

```bash
cd stat
wget https://huggingface.co/nyu-visionx/RAEv2-models/resolve/main/stage1/imagenet/dinov3l-k7/stats.pt \
    -O dinov3l_k7_imagenet1k_stats.pt
```

### 2. 修改训练脚本路径

编辑 `sh/train_pixel_raev2.sh`，设置：
- `IMAGENET` - ImageNet 数据集路径
- `DINOV3_REPO` - DINOv3 源码目录（包含 hubconf.py）

### 3. 启动训练

```bash
bash sh/train_pixel_raev2.sh
```

---

## 推理（需要 RAEv2 环境）

推理时需要从 RAEv2 Stage 1 采样 latent。两种方案：

### 方案 1：在 RAEv2 环境中预先采样 latent

```bash
# 1. 在 RAEv2 环境中采样 latent
cd /path/to/RAEv2
python scripts/stage1/sample.py \
    --encoder dinov3mls-vit-l16[layers=11.13.15.17.19.21.23] \
    --decoder pretrained_models/stage1/imagenet/dinov3l-k7/decoder.pt \
    --normalization_stat_path pretrained_models/stage1/imagenet/dinov3l-k7/stats.pt \
    --num_samples 50000 \
    --output latents.pt

# 2. 在 Pixel JiT 环境中读取 latent 并生成图像
cd /path/to/jit_cond_dino_two_stage_1122_raev2
python sample_with_precomputed_latents.py \
    --checkpoint checkpoints/.../checkpoint-160.pth \
    --latents /path/to/latents.pt \
    --output samples/
```

### 方案 2：集成 RAEv2 decoder（待实现）

将 RAEv2 的 ViTMAEDecoder 复制到本项目，实现端到端采样。

---

## 预期结果

| 配置 | FID | 对比 pred 4.71 | 理由 |
|------|-----|---------------|------|
| **DINOv3-L k=7 + clean** | **4.2–4.4** | **−0.3 to −0.5** | 更强 encoder + 完整实例信息 |

**优势**：
- **无需 Stage 1 训练**：节省 E160 训练成本
- **更强 encoder**：DINOv3-L k=7 比 DINOv3-B 单层更强
- **最强实例信息**：训练时用 clean latent（100% 实例信息）
- **更大容量**：1024 维 vs 768 维

---

## 文件清单

```
jit_cond_dino_two_stage_1122_raev2/
├── stage1/
│   └── dinov3_k7_encoder.py       # DINOv3-L k=7 encoder 实现
├── stat/
│   ├── DOWNLOAD.md                # 统计文件下载说明
│   └── dinov3l_k7_imagenet1k_stats.pt  # 需从 HuggingFace 下载
├── sh/
│   └── train_pixel_raev2.sh       # 训练脚本
├── denoiser.py                    # Pixel JiT（使用 clean latent）
├── main_jit.py                    # 训练主程序
├── checkpoint_io.py               # Checkpoint I/O（简化版）
└── README_raev2.md                # 本文档
```

---

## 实现状态

### ✅ 已完成

- DINOv3-L k=7 encoder 实现
- Denoiser 修改（使用 clean latent）
- main_jit.py 参数调整
- checkpoint_io.py 简化
- 训练脚本

### 🔄 H200 上需要做的

1. 下载统计文件（见上方步骤 1）
2. 修改训练脚本路径（见上方步骤 2）
3. 启动训练（见上方步骤 3）

代码已全部完成，可在 H200 上下载统计文件后启动训练。
