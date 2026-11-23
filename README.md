# jit_cond_ablation 实验目录说明

ImageNet 256 上研究将 DINO patch tokens 注入 JiT patch-level AdaLN 的系列消融实验。

---

## 目录结构

| 目录 | 对应表 | 说明 |
|---|---|---|
| `JiT_cond` | 表 1 | clean 图像 DINO patch tokens + patch-level AdaLN，oracle 参考上界 |
| `JiT_cond_cls` | 表 1 | clean 图像 DINO CLS token + block-level AdaLN，已排除此路线 |
| `jit_cond_dino_two_stage_1122` | 表 2、3 | noised / pred 两种 Stage 2 条件模式（见下方 bug 说明） |
| `jit_cond_dino_two_stage_1122_pred_drop` | 表 3（重跑） | 修复 P1 后的 pred 模式，**重跑表 3 应使用此目录** |
| `jit_cond_dino_two_stage_1122_ddim_recon_noflip` | 表 3 未来 | Stage 1 加 MSE 重建约束 + noflip，Stage 2 cached_recon（尚未启动）|
| `jit_cond_dino_two_stage_1122_ddim_recon_noflip_pred_drop` | 表 3 未来 | noflip 版同步修复 P1–P4，Stage 2 启动时使用此目录 |

---

## 2026-11-24 Bug 修复记录

### P1（影响已有实验）：pred 模式 label dropout 缺失

**受影响目录**：`jit_cond_dino_two_stage_1122`，仅限 `condition_mode_pixel=pred` 的 Stage 2 训练。

**问题描述**：`denoiser.py` 的 `prepare_grounding_pixel` 在 pred 模式下直接用原始 `labels` 调用冻结 DinoDiT：

```python
# 旧（有 bug）
z_cond = self.dino_net(z_cond, t_d_p, labels)
```

训练时 Pixel JiT 只见过 DinoDiT 在有条件（真实类别）下输出的 token。但推理时 DinoDiT CFG 实际输出是 `v_uncond + w*(v_cond − v_uncond)` 的混合，token 分布随 CFG scale 变化。这个 train-test gap 导致已报告的表 3 FID 偏高。

**修复**：用独立 dropout 的 `labels_for_dino` 调用 DinoDiT，让 Pixel 训练同时覆盖有条件和无条件 token 分布：

```python
# 新（修复版）
labels_for_dino, _ = self.drop_labels_with_mask(labels)
z_cond = self.dino_net(z_cond, t_d_p, labels_for_dino)
```

Pixel 侧的 `labels_dropped` 和 `labels_for_dino` 是两次独立 dropout，互不干扰。

**受影响的具体数据点（表 3，均需重跑）**：

| E_D | E_P | CFG | 旧 FID | 状态 |
|-----|-----|-----|--------|------|
| 80 | 20 | 2.9 | 28.61 | 旧版，低优先级 |
| 80 | 40 | 2.9 | 6.93 | 旧版，低优先级 |
| 80 | 60 | 2.9 | 5.64 | 旧版，低优先级 |
| 80 | 80 | 2.9 | 5.58 | 旧版，低优先级 |
| 80 | 80 | 2.3 | 5.18 | 旧版，低优先级 |
| **160** | **80** | **2.9** | **5.43** | **需重跑** |
| **160** | **80** | **2.1** | **4.78** | **需重跑** |
| **160** | **160** | **2.9** | **5.25** | **需重跑** |
| **160** | **160** | **2.1** | **4.71** | **需重跑，当前最佳** |

noised 模式（表 2）不受影响，Stage 1 训练不受影响。

**重跑方案**：DinoDiT E160 checkpoint 可直接复用（`jit_cond_dino_two_stage_1122/checkpoints/stage1-dinov3b-dinodit-b-e160`），只需用 `jit_cond_dino_two_stage_1122_pred_drop` 重训 Pixel JiT 160 epoch。

---

### P2（未触发，预防性修复）：logit_normal 分支复用 Stage 1 时间参数

**受影响目录**：所有已跑实验均使用 `--t_schedule_dino_pixel uniform`，P2 从未触发，无历史结果受影响。

**问题描述**：`logit_normal` 分支错误复用 `t_mean_dino/t_std_dino`（Stage 1 参数，默认 −1.2/1.0），对应全时间范围分布，与 Stage 2 目标区间 [0.75, 1) 不匹配。

**修复位置**：`jit_cond_dino_two_stage_1122_ddim_recon_noflip_pred_drop`，新增 `--t_mean_dino_pixel`（默认 1.95）和 `--t_std_dino_pixel`（默认 0.5）两个独立参数。

---

### P3（预防性）：resume 校验补全新参数

`checkpoint_io.py` 的 `validate_metadata` 新增对 `t_mean_dino_pixel`、`t_std_dino_pixel` 的续训校验，旧 checkpoint 自动补默认值以透明兼容。仅影响 `jit_cond_dino_two_stage_1122_ddim_recon_noflip_pred_drop`。

---

### P4（文档性）：`_velocity_dino` 无 `@torch.no_grad()` 说明

`stage1/denoiser_dino.py` 的 `_velocity_dino` 故意不加 `@torch.no_grad()`，原因是 `instance_reconstruction_loss` 中的学生解码需要保留梯度。在代码中加注释说明，防止与原版 1122 合并时误加装饰器。

---

## 实验约定

- `E_D` / `E_P`：DinoDiT / Pixel JiT 已完成训练 epoch 数
- FID 越低越好，IS 越高越好，均基于 50,000 张生成图
- 采样默认 DINO Heun-25 + Pixel Heun-50，CFG interval [0.1, 1]
- noised 模式：Stage 2 条件为加噪 clean latent（`t_d_p ~ U[0.75,1)`）
- pred 模式：Stage 2 条件为冻结 DinoDiT 一步预测结果
- cached_recon 模式：Stage 2 条件为训练集预存的 EMA1 25 步重建特征
- 采样时均不使用真实参考图；表 1 / 表 2 中的 clean-image 行为 oracle 参考
