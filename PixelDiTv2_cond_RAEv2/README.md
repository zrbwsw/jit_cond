# PixelDiTv2_cond_RAEv2

使用 JiT 的训练与评估框架，在 ImageNet 256×256 上训练 PixelDiT2-H/16。输入是 RGB 噪声图，输出预测 clean RGB。训练与生成共用同一个 patch-level AdaLN 接口。

| 阶段 | Grounding 来源 | Pixel 条件 |
| --- | --- | --- |
| 训练 | 同一张 clean crop 经冻结的 DINOv3-L/16 k=7，再用 RAEv2 的 mean、var 标准化 | 每个 patch 使用自己的 grounding，加上 Pixel 时间和类别 embedding |
| 生成 | 冻结的 RAEv2 Stage 2 DDT EMA 从独立高斯噪声和类别采样 normalized latent | 每个样本取一次 latent，复用于整条 Pixel Heun 轨迹 |

两路 latent 的形状都是 `[B,1024,16,16]`。按空间顺序展开为 `[B,256,1024]`，共用一个 `Linear(1024,1280)`。默认 `condition_mode_pixel=clean`，模型将 grounding、Pixel 时间和类别三项相加并做原版 PixelDiT2 的外层 SiLU。训练时只有类别按 0.1 概率替换为 null，即使替换了类别，每个 patch 仍保留 grounding。采样沿用原 PixelDiT2 CFG，无条件分支使用 null 类别并跳过 grounding。因此训练中出现的 null 类别输入仍比采样的无条件分支多一份语义信息。

设置 `--condition_mode_pixel noised` 后，训练按 `t_dino_pixel ~ U[1-noise_beta_dino_pixel,1)` 对 clean latent 加噪，并额外加入对应的 DINO 时间 embedding；默认 beta 为 `0.25`。采样时 conditional 分支统一使用 `t_dino=1`，unconditional 分支仍跳过 grounding 和该时间条件。clean 模式不会创建或加载这套额外 embedding，旧 clean checkpoint 可以继续恢复。

设置 `--condition_mode_pixel pred` 后，训练会将 noised latent 交给冻结的 RAEv2 DDT-XL EMA，取 full branch 的单步 clean x-prediction，不使用 CFG/IG；Pixel 使用 `t_dino=1`。RAEv2 的时间方向为 `x_t=(1-t)*clean+t*noise`，所以 DDT 接收 `t_rae=1-t_dino_pixel`。只有 pred 训练需要加载 DDT checkpoint 和配置，采样仍使用官方完整 RAEv2 sampler。

## 训练与采样设置

- Pixel 主干使用 32 层、16 个 head、1280 hidden、16×16 像素 patch，保留原 patch embedding、二维 RoPE 和中间层 0.2 projection dropout。
- 默认损失为 `L_FM + 0.5 × L_REPA`。FM 在 velocity 空间计算。REPA teacher 是冻结的 DINOv2-B/14，同一张 clean crop 经原版预处理后提供 256 个目标 tokens，每个 768 维。第 8 层输出经三层 MLP 后对齐 teacher，损失为平均 `1 − cosine_similarity`。
- REPA 取 `patch_blocks[7]`，此处正好是 256 个 image tokens。32 个 context tokens 到 `patch_blocks[8]` 才插入。若显式改到后面的层做 REPA，代码会先移除 context tokens。
- 默认训练 160 epoch。H200 默认使用 8 卡，每卡 batch 128，每个 batch 执行一次 optimizer 更新，global batch 为 1024。AdamW 使用 betas `(0.9,0.95)`、weight decay 0，前 6255 次 optimizer 更新线性 warmup 到 `2e-4`，之后保持恒定。采样使用 decay 0.9999 的 Pixel EMA。
- RAEv2 sampler 从官方 `configs/stage2/sampling/imagenet-dinov3l-k7.yaml` 读取参数，默认 100 步、CFG 1.0、IG 1.78，IG 区间为 `[0.1,1]`。RAEv2 按自己的时间约定从 1 积分到 0，Pixel 从 0 积分到 1。clean 模式不向 Pixel 条件加入语义时间；noised 模式在训练时加入实际 `t_dino_pixel`，pred 和采样统一为 1。Pixel 使用 Heun 50 步，最后一步 Euler。Pixel CFG 2.4 只在 `0.125 < t < 0.9` 时生效。

## H200 入口

在 GPU 节点激活现有环境，从本目录执行。脚本默认单机 8 卡，支持 `PYTHON_BIN`、`CUDA_VISIBLE_DEVICES`、`NPROC_PER_NODE` 和额外的 Python CLI 参数。两个脚本均可附加 `--dry-run` 查看命令。

```bash
bash sh/train_pixel.sh
bash sh/train_pixel.sh --resume /path/to/checkpoint-last.pth
PIXEL_CHECKPOINT=/path/to/checkpoint-160.pth bash sh/sample_pixel.sh

# 可选：训练时使用加噪 grounding；GPU2 脚本会自动使用独立的 noised 输出目录
CONDITION_MODE=noised NOISE_BETA_DINO_PIXEL=0.25 bash sh/train_pixel_gpu2.sh

# 可选：加噪后先经过冻结的 RAEv2 DDT-XL 单步 clean prediction
CONDITION_MODE=pred NOISE_BETA_DINO_PIXEL=0.25 bash sh/train_pixel_gpu2.sh

# noised/pred checkpoint 采样时也必须指定相同模式
PIXEL_CHECKPOINT=/path/to/checkpoint-160.pth bash sh/sample_pixel.sh --condition_mode_pixel noised
```

训练读取 ImageNet 的 `train/`、本地 DINOv3 源码和权重、k=7 统计值，以及 DINOv2-B teacher 权重；pred 模式还需要 RAEv2 DDT checkpoint 和配置。默认训练脚本仅训练和存 checkpoint；追加 `--online_eval --eval_freq 20` 后会在完成第 20、40、…、160 个 epoch 时采样评估。独立采样读取 Pixel checkpoint、同一份统计值、RAEv2 源码、官方 YAML 和 DDT 权重。默认计算 50K FID/IS；传 `--samples_only --num_images 16` 可改为保留 16 张 PNG。

| 资源 | 环境变量 |
| --- | --- |
| ImageNet 根目录 | `IMAGENET_ROOT` |
| DINOv3 源码、DINOv3-L 权重、DINOv2-B 权重 | `DINOV3_REPO`、`DINO_ENCODER_WEIGHTS_PATH`、`REPA_ENCODER_WEIGHTS_PATH` |
| RAEv2 DINOv3-L k=7 的 stats.pt | `DINO_STAT_PATH` |
| RAEv2 源码根目录、DDT checkpoint、采样 YAML | `RAEV2_ROOT`、`RAEV2_CHECKPOINT`、`RAEV2_CONFIG` |
| JiT ImageNet-256 FID 统计目录 | `FID_STATS_ROOT` |

Pixel checkpoint 包含主干、grounding Linear、REPA projection、可训练参数 EMA、optimizer、已完成的 epoch 数及统计文件的 SHA-256；noised/pred 模式还包含 DINO time embedder。续训和独立采样都会校验统计指纹及 condition 模式。默认输出目录为 `checkpoints/raev2-dinov3l-k7-pixeldit2-h16-e160/`，noised/pred 分别写入带对应模式后缀的独立目录。

训练与评估入口为 [main_jit.py](main_jit.py)，grounding、loss 和 Heun 位于 [denoiser.py](denoiser.py)，主干和 REPA 分别在 [pixeldit2_c2i.py](pixeldit2_c2i.py) 和 [repa.py](repa.py)。在现有 PyTorch 环境运行 `python -m pytest -q tests` 可用合成张量验证接口、梯度、EMA、checkpoint 和采样分支。
