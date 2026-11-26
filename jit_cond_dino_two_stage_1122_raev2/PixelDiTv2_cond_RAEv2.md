# PixelDiTv2_cond_RAEv2 实现方案

## 已确定的设计要求

1. **目标代码目录为 `PixelDiTv2_cond_RAEv2`。**训练框架使用 JiT；Pixel-Diffusion-H 的模型结构使用 PixelDiT2-H，包括 REPA 的逻辑、loss 形式和 encoder。替换的是 PixelDiT2 的完整 grounding 分支，即从 condition 输入处完整替换，不保留 `x_t -> frozen DINO -> PixelDiT2 timestep-conditioned projection`，因此不讨论这条旧分支内部的输入归一化等问题。
2. **不需要 `t_dino_pixel`。**Stage 2 training 输入 clean latent，Stage 2 inference 输入 RAEv2 采样得到的 latent；两者都位于 `t=1` 的接近干净端，不为 condition 额外增加恒定的 DINO 时间 embedding。Pixel 自身的 `t` 和类别 `y` 条件保持不变。
3. **grounding 只增加一个线性维度映射。**DINO feature 或 RAE DDT 输出先转成 patch tokens，再经一个 Linear 映射到 PixelDiT2-H 的 hidden size，使其与 `t`、`y` embedding 的维度一致。
4. **默认训练 PixelDiT2-H 160 epoch。**训练中不使用 grounding-token dropout，grounding 始终保留。
5. **默认只支持 ImageNet `256×256`。**本期不处理分辨率缩放，也不处理 H/32 或 512 分辨率下的 grounding 网格变化。
6. **PixelDiT2 主干的结构、超参数和采样参数保持不变。**除完整 grounding 分支被替换外，不改变 patch embedding、时间编码、位置编码、Transformer block、context token、REPA 和 Pixel Heun sampling。
7. **RAEv2 全部冻结。**训练不调用 RAEv2 Stage 2 DDT；clean grounding 由冻结的 DINO feature encoder 提供。采样时才加载冻结的 RAEv2 Stage 2 DDT，默认采样配置与 RAEv2 官方配置一致。

## 1. 术语和源码路径

以下源码引用均相对于工作根目录，即同时包含 `jit_cond_ablation/`、`Pixel_DIffusion/` 和 `RAEv2/` 的目录。

“训练用不到 RAEv2”在本文中特指**不加载、不调用 RAEv2 Stage 2 DDT**。训练仍需要冻结的 RAEv2 DINOv3-L k=7 feature encoder，将真实 clean image 编码到与 RAEv2 Stage 2 相同的 normalized latent space。

主要参考实现：

- JiT 训练与评估框架：`jit_cond_ablation/jit_cond_dino_two_stage_1122_raev2/main_jit.py`、`engine_jit.py`、`checkpoint_io.py`、`util/`
- 当前 clean RAEv2 grounding：`jit_cond_ablation/jit_cond_dino_two_stage_1122_raev2/denoiser.py`
- RAEv2 DINOv3-L k=7 encoder：`jit_cond_ablation/jit_cond_dino_two_stage_1122_raev2/stage1/dinov3_k7_encoder.py`
- RAEv2 Stage 2 latent sampler：`jit_cond_ablation/jit_cond_dino_two_stage_1122_raev2/raev2_latent_sampler.py`
- PixelDiT2-H 模型：`Pixel_DIffusion/PixelDiT/pixdit_core/pixeldit2_c2i.py`
- PixelDiT2 基础模块：`Pixel_DIffusion/PixelDiT/pixdit_core/modules.py`
- PixelDiT2-H/16@256 配置：`Pixel_DIffusion/PixelDiT/c2i/configs/pixeldit2_h16_in256.yaml`
- PixelDiT2 REPA、flow loss 和 sampler：`Pixel_DIffusion/PixelDiT/c2i/src/grounded_diffusion.py`
- RAEv2 官方采样配置：`RAEv2/configs/stage2/sampling/imagenet-dinov3l-k7.yaml`

## 2. 总体数据流

### 2.1 Stage 2 training

```text
clean image x_0，范围 [-1, 1]
  -> x_0 * 0.5 + 0.5
  -> frozen DINOv3-L k=7 encoder
  -> z-score normalized clean latent [B, 1024, 16, 16]
  -> flatten(2).transpose(1, 2)
  -> clean grounding tokens [B, 256, 1024]
  -> Linear(1024, 1280)
  -> PixelDiT2-H patch condition

x_0、t、y
  -> PixelDiT2-H RGB flow matching
  -> REPA DINOv2-B teacher + PixelDiT2 第 8 层特征
  -> L = L_FM + 0.5 * L_REPA
```

训练过程中不实例化 RAEv2 Stage 2 DDT，不调用 RAE decoder，不对 grounding 加噪，也不使用 `t_dino_pixel`。

### 2.2 Stage 2 inference

```text
class labels y
  -> frozen RAEv2 Stage 2 DDT EMA，官方默认采样配置
  -> sampled normalized latent [B, 1024, 16, 16]
  -> grounding tokens [B, 256, 1024]
  -> 与训练共用的 Linear(1024, 1280)
  -> PixelDiT2-H EMA
  -> 原 PixelDiT2 Heun-50 sampling
  -> RGB images
```

每个样本的 RAEv2 latent 只采样一次，得到的 grounding tokens 在该样本的全部 Pixel sampling steps 中固定复用。RAEv2 DDT 不参与 Pixel 的反向传播，也不进入 Pixel optimizer 或 EMA。

## 3. 保持不变的 PixelDiT2-H/16 配置

除 grounding 分支外，模型以 `Pixel_DIffusion/PixelDiT/pixdit_core/pixeldit2_c2i.py` 和 `Pixel_DIffusion/PixelDiT/c2i/configs/pixeldit2_h16_in256.yaml` 为准。

| 项目 | 保留配置 |
| --- | --- |
| 输入与输出 | 直接预测 `[-1,1]` RGB，`[B,3,256,256]`；不使用 v1 PiT |
| Patch | `patch_size=16`，得到 `16×16=256` 个 image tokens |
| Hidden / heads / depth | `hidden_size=1280`、`num_heads=16`、`depth=32` |
| Context | `in_context_len=32`、`in_context_start=8` |
| Patch embedding | `unfold + PatchTokenEmbedder`，不改写为 JiT bottleneck |
| 时间编码 | 保留 PixelDiT2 `TimestepConditioner`，包括其频率实现 |
| 位置编码 | 保留 PixelDiT2 动态二维 RoPE 和 context position embedding |
| Dropout | `attn_dropout=0.0`、`proj_dropout=0.2` |
| FFN | `mlp_hidden_multiple=0`，保留原 SwiGLU 实现 |
| Condition 激活 | `single_silu_cond=false`，保留外层 `F.silu(c)` 行为 |
| Pixel flow | `p_mean=-0.8`、`p_std=0.8`、`noise_scale=1.0`、`t_eps=0.05` |
| Pixel sampler | Heun 50 steps，guidance `2.4`，interval `[0.125,0.9]` |
| Optimizer | AdamW，LR `2e-4`，betas `(0.9,0.95)`，weight decay `0` |
| EMA | 使用 `0.9999` EMA 权重进行采样 |
| 训练轮数 | 按本方案改为 160 epoch；其余训练超参数不因更换框架而改变 |

JiT 默认 `batch_size=128`、8 GPU 对应 global batch 1024；`blr=5e-5` 经 `global_batch/256` 缩放后得到 LR `2e-4`，与上述 PixelDiT2 配置一致。启动脚本应显式设置 `--proj_dropout 0.2`，不能沿用 JiT 的 `0.0` 默认值。

## 4. 目标目录和文件职责

目标目录：

```text
jit_cond_ablation/PixelDiTv2_cond_RAEv2/
```

建议按以下职责组织，避免把 PixelDiT2 Lightning 框架与 JiT 训练框架混合：

- `main_jit.py`：沿用 JiT 的训练、恢复和评估入口；默认 `epochs=160`，模型固定为 PixelDiT2-H/16。
- `denoiser.py`：组合 clean grounding、PixelDiT2 flow loss、REPA loss 和两阶段采样。
- `pixeldit2_c2i.py`：PixelDiT2-H 主干；只改写 grounding 输入和 projection 接口。
- `modules.py`：保留 PixelDiT2 的 patch embed、time/position encoding、attention、FFN、AdaLN 和 final layer。
- `grounding_raev2.py`：提供 clean latent 编码、latent-to-token 转换和 `Linear(1024,1280)`。
- `repa.py`：保留 PixelDiT2 的 REPA teacher、projection、hook 和 cosine loss。
- `raev2_latent_sampler.py`：复用当前 RAEv2 官方 Stage 2 EMA sampler 封装。
- `stage1/dinov3_k7_encoder.py`：复用当前 DINOv3-L k=7 encoder 和 latent normalization。
- `engine_jit.py`、`checkpoint_io.py`、`resource_paths.py`、`util/`：沿用 JiT 的训练、EMA、checkpoint、日志和评估设施。
- `sh/train_pixel.sh`、`sh/sample_pixel.sh`：训练与采样启动脚本，提供 `--dry-run`。
- `tests/`：覆盖 condition、REPA、冻结、采样次数和 checkpoint roundtrip。

本期以新训练 160 epoch 为主，不处理官方 PixelDiT2-H checkpoint 的部分加载，也不处理 JiT-B checkpoint 向 H 主干的转换。

## 5. 模型侧修改

### 5.1 完整移除旧 grounding 分支

在目标 `pixeldit2_c2i.py` 中移除：

- `GroundingConditioner` 及其 import；
- `dino_conditioner` 子模块；
- `grounding_encoder`、`grounding_proj_blocks`、`grounding_proj_heads`、`pretrained_encoder` 等旧参数；
- forward 内部的 `self.dino_conditioner(x, t)`；
- `return_grounding_tokens` 中对旧 encoder 的依赖。

不再使用 `Pixel_DIffusion/PixelDiT/pixdit_core/grounding.py`。Pixel 输入 `x_t` 只进入原 PixelDiT2 patch embed 和 RGB flow-matching 主干。

### 5.2 新的 grounding 接口

新增：

```python
self.grounding_proj = nn.Linear(1024, hidden_size, bias=True)
```

PixelDiT2-H 的 `hidden_size=1280`。forward 显式接收：

```python
grounding_tokens: torch.Tensor  # [B, 256, 1024]
```

在 condition 构造前检查 batch、token 数和 feature dim，随后执行：

```python
grounding = self.grounding_proj(grounding_tokens)  # [B,256,1280]
c = grounding + t_emb.unsqueeze(1) + y_emb
if not self.single_silu_cond:
    c = F.silu(c)
```

这与原 PixelDiT2 的 condition 相加方式一致，但 `grounding` 不再依赖 `x_t`，也不经过旧 `TimestepProjection`。不增加 `t_dino_pixel`，不增加额外的 grounding Transformer block。

原 context 逻辑保持不变：context token 的 condition 仍由 `c.mean(dim=1)` 扩展得到，image patch 使用逐 patch 的 `c`。

### 5.3 Null condition 与训练 dropout

训练时始终调用新的 grounded condition，不生成 grounding drop mask，也不随机清零 `grounding_tokens`。

类别 dropout 仍保留原概率 `0.1`，用于 classifier-free guidance。类别被 drop 时，语义 grounding 仍保留，这与“训练中 grounding 始终保留”的要求一致。

采样器保持原 PixelDiT2 逻辑：conditional 分支使用类别和 RAEv2 grounding；unconditional 分支使用 null class，并沿用原 `skip_grounding=True` 的 `_null_condition`。因此，“训练始终保留 grounding”并不意味着采样 unconditional 分支也保留 grounding。

这会形成一个需要记录的训练/采样差异：训练不覆盖“无类别且无 grounding”的 null 分支。本期不通过增加 grounding dropout 来改变既定训练设置，但必须在验证和结果分析中明确这一点。

## 6. 训练侧修改

### 6.1 Clean grounding

沿用 JiT 数据流程，图像进入模型前为 `[-1,1]`。clean encoder 调用前执行：

```python
clean_latent = dino_encoder.encode(x_0 * 0.5 + 0.5)
grounding_tokens = clean_latent.flatten(2).transpose(1, 2).contiguous()
```

encoder 输出必须为 `[B,1024,16,16]`，并使用与 RAEv2 Stage 2 相同的 `stats.pt`、z-score normalization 和 k=7 特征聚合方式。训练和采样应校验相同的 latent statistics fingerprint。

这里只处理真实 clean image 的 encoder 输入范围，不涉及已删除的旧 `x_t` grounding 分支。

### 6.2 Flow matching

保留 `Pixel_DIffusion/PixelDiT/c2i/src/grounded_diffusion.py` 的形式：

```text
t ~ sigmoid(N(-0.8, 0.8))
noise = N(0, I)
x_t = t * x_0 + (1 - t) * noise
v_target = (x_0 - x_t) / clamp(1 - t, min=0.05)
v_pred = (x_pred - x_t) / clamp(1 - t, min=0.05)
L_FM = mean((v_target - v_pred)^2)
```

JiT 外层的 `train_one_epoch` 只负责数据、autocast、backward、optimizer、EMA 和日志，不能再对同一批图像重复采样时间或重复加噪。

### 6.3 REPA

完整保留 PixelDiT2 的 REPA 设置和计算方式：

- `repa_weight=0.5`；
- `repa_align_layer=8`，在 `patch_blocks[7]` 注册 forward hook；
- 第 8 层已经插入 32 个 context tokens，计算 REPA 前按原逻辑移除这些 tokens；
- 使用 frozen DINOv2-B/14 teacher，`bf16_weights=true`、`dinov2_pos_embed=true`；
- teacher 输入为 `(x_0 + 1) / 2`，目标分辨率和 token 对齐方式沿用 `REPATeacher`；
- 保留 `_match_token_grid`，尽管本期 256 分辨率下应为直接相等。

REPA projection 保留原三层结构：

```text
Linear(1280,1280) -> SiLU
Linear(1280,1280) -> SiLU
Linear(1280,768)
```

loss 为：

```text
L_REPA = mean(1 - cosine_similarity(proj(h_block_8), z_dinov2))
L = L_FM + 0.5 * L_REPA
```

REPA projection 是可训练参数，必须进入 optimizer、EMA 和 checkpoint；DINOv2 teacher 始终 frozen、eval、`no_grad`。

## 7. 采样侧修改

### 7.1 RAEv2 semantic sampling

复用 `raev2_latent_sampler.py`，不重新实现 RAEv2 transport、sampler 或 guidance。默认使用：

- RAEv2 Stage 2 `ema` checkpoint；
- `RAEv2/configs/stage2/sampling/imagenet-dinov3l-k7.yaml`；
- latent shape `[1024,16,16]`；
- 100 steps；
- CFG `1.0`；
- IG `1.78`，IG interval `[0.1,1]`。

RAEv2 sampler 应作为 Pixel 模型外部的冻结服务，按每个 worker/rank 初始化。不要把 DDT 挂到 PixelDiT2 的 `nn.Module` 中，避免它被 optimizer、EMA、模型复制或 checkpoint 误处理。

### 7.2 Pixel sampling

采样一个 batch 时：

1. 根据 labels 调用一次 RAEv2 sampler。
2. 将输出转换为 `[B,256,1024]`，后续由 Pixel 内的共享 Linear 映射到 1280。
3. 将同一份 tokens 传给 Pixel 在线模型或 EMA 模型。
4. 按原 PixelDiT2 Heun-50、guidance `2.4`、interval `[0.125,0.9]` 完成 RGB sampling。
5. conditional 和 unconditional 分支共享同一份采样 latent；unconditional 只通过原 null-condition 路径屏蔽 grounding，不能重新采样另一份 latent。

Pixel sampler 的步数、guidance、noise scale、`t_eps` 和最后一步 Euler 均不改用 JiT 参数。

## 8. 参数、checkpoint 与冻结边界

### 可训练参数

- PixelDiT2-H 主干；
- `grounding_proj: Linear(1024,1280)`；
- REPA projection。

### 冻结且不进入 optimizer/EMA 的参数

- DINOv3-L k=7 clean feature encoder；
- DINOv2-B REPA teacher；
- RAEv2 Stage 2 DDT。

### Checkpoint

沿用 JiT checkpoint 和 resume 机制，至少保证：

- Pixel 主干、`grounding_proj` 和 REPA projection 完整保存并恢复；
- EMA 只包含可训练参数；
- 保存 PixelDiT2-H 配置、REPA 配置、RAEv2 encoder 类型和 latent statistics fingerprint；
- RAEv2 DDT 不写入 Pixel checkpoint，采样时由 `raev2_checkpoint` 和 `raev2_config` 单独指定；
- 只允许相同 `PixelDiTv2_cond_RAEv2` 格式进行严格 resume，不与旧 JiT 或 PixelDiT2 checkpoint 混用 optimizer。

## 9. 启动参数整理

删除或不再暴露：

- `condition_mode_pixel`；
- `noise_beta_dino_pixel`、`t_schedule_dino_pixel`；
- `use_t_dino`、`t_dino_pixel`；
- 旧 `grounding_encoder`、`grounding_proj_blocks`、`grounding_proj_heads`；
- 所有 training grounding-drop 参数；
- 分辨率插值相关参数。

保留或新增：

- `--model PixelDiT2-H/16`；
- `--epochs 160`；
- `--proj_dropout 0.2`；
- clean encoder 的 weights、repo、`dino_stat_path`；
- `raev2_checkpoint`、`raev2_config`，仅采样必需；
- REPA weight、align layer、teacher 和 projection 配置；
- Pixel Heun、guidance 和 interval 参数，其默认值与 PixelDiT2-H/16@256 完全一致。

## 10. 验证计划

### 单元测试

1. **形状**：`[2,256,1024] -> grounding_proj -> [2,256,1280]`，最终 RGB 输出为 `[2,3,256,256]`。
2. **Condition 数值**：手工计算 `grounding + t + y` 和外层 SiLU，与模型 condition 一致；不出现额外 DINO time embedding。
3. **旧分支移除**：mock 旧 `GroundingConditioner`，确认 forward 不会调用；训练不会实例化 RAEv2 DDT。
4. **训练 dropout**：多次 forward 均保留 grounding；只有类别 label 可以按概率变为 null class。
5. **REPA**：hook 确实位于第 8 层，context tokens 被正确移除，teacher 输出为 768 维，projection 输出与 teacher 对齐，总 loss 严格为 `fm + 0.5 * cosine`。
6. **梯度和 EMA**：Pixel、`grounding_proj`、REPA projection 有梯度并进入 EMA；两个 frozen encoder 无梯度且不进入 EMA。
7. **采样次数**：mock RAEv2 sampler，确认每个生成 batch 只调用一次；所有 Pixel steps 使用相同 tokens。
8. **CFG 分支**：conditional 使用 tokens；unconditional 使用原 null condition；不得为两个分支生成不同 semantic latent。
9. **Checkpoint roundtrip**：Pixel、grounding projection、REPA projection、optimizer 和 EMA 均能恢复。
10. **配置回归**：检查 H/16、REPA 和 Heun 参数未被 JiT 默认值覆盖。

### 小规模端到端验证

1. 单步真实训练：同时输出有限的 `fm_loss`、`cos_loss` 和总 loss，确认反传正常。
2. 小批量生成：完成一次真实 RAEv2 DDT + PixelDiT2-H 采样，检查 latent、RGB 形状和数值有限性。
3. CFG smoke test：分别运行 conditional/unconditional 路径，确认不存在 token batch 对齐错误。
4. 多卡 smoke test：确认每个 rank 的 RAEv2 sampler 独立初始化，label、latent 和生成图像顺序一致。
5. 在上述检查通过后，再启动 160 epoch 正式训练和 50K 评估。

## 11. 明确不在本期范围内

- 不恢复或改造旧 `x_t` grounding；
- 不引入 `t_dino_pixel`；
- 不进行 grounding corruption；
- 不进行 training grounding dropout；
- 不改变 PixelDiT2-H 主干、REPA 或 Pixel sampler；
- 不处理 512、H/32 或任意分辨率插值；
- 不在训练中调用 RAEv2 Stage 2 DDT；
- 不使用 RAE decoder 生成最终 RGB；
- 不处理官方 PixelDiT2-H checkpoint 的部分加载；
- 不包含 v1 PiT。

本文档仅描述目标实现方案，不表示 `PixelDiTv2_cond_RAEv2` 代码已经完成或通过上述验证。
