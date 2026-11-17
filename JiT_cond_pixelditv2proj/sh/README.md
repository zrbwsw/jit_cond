# Noisy grounding 训练与采样脚本

脚本默认 `GPU_HOST=local`，在当前机器前台运行。默认8卡、每卡 batch128、600 epoch，实验名为 `JiT-B16-DINOv3B-Noisy-PixelDiT2Proj-B600`。`--dry-run` 只打印命令，不启动训练、采样或 SSH。

```bash
bash sh/train_jit_b16_grounding_8gpu.sh --dry-run
bash sh/train_jit_b16_grounding_8gpu.sh
```

采样从随机噪声开始，输入条件为类别标签。无需原图、ImageNet 数据目录或验证集配对；计算 FID/IS 使用本地预计算统计文件。

```bash
CHECKPOINT=/path/to/new-variant/checkpoint-last.pth \
  bash sh/sample_jit_b16_grounding_8gpu.sh --dry-run
```

确认后去掉 `--dry-run`。默认 checkpoint 为本版本训练目录的 `checkpoint-last.pth`，缺失时直接报错。旧 `JiT_cond` checkpoint 不可直接恢复到本版本。

## 常用环境变量

| 变量 | 默认值或含义 |
| --- | --- |
| `PYTHON_BIN` | 优先 `$HOME/anaconda3/envs/irepa/bin/python`，其次原共享环境，再使用 PATH 中的 python |
| `GPU_HOST` | `local`；显式设置 SSH alias/host 才转交远端 |
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3,4,5,6,7` |
| `GPUS` | 默认等于可见 GPU 数量，不能大于可见卡数 |
| `MODEL` / `IMG_SIZE` / `NOISE_SCALE` | `JiT-B/16` / `256` / `1.0` |
| `PRETRAIN_MODELS_ROOT` | 本机存在时优先 `/data0/pretrain_models`，否则由 `resource_paths.py` 解析 |
| `DINO_REPO` 或 `DINOV3_REPO` | 本地含 `hubconf.py` 的 DINOv3 源码，默认由 `resource_paths.py` 搜索 |
| `DINO_WEIGHTS` 或 `DINO_WEIGHTS_PATH` | 默认 DINOv3-B/16 权重文件 |
| `FID_STATS_ROOT` | 默认项目父目录中的 `fid_stats` |
| `IMAGENET_PATH` 或 `IMAGENET_ROOT` | 仅训练使用；本机存在时默认 `/data0/datasets/imagenet` |
| `GROUNDING_PROJ_BLOCKS` | `1` |
| `GROUNDING_PROJ_HEADS` | 不传时沿用 JiT 的 head 数；B 为12 |
| `GROUND_DROP_PROB` | `0.0`，仅训练使用 |
| `BATCH_SIZE` / `EPOCHS` / `BLR` | `128` / `600` / `5e-5` |
| `OUTPUT_DIR` | `checkpoints/$EXPERIMENT` |
| `RESUME` | 默认 `OUTPUT_DIR`；设为空字符串可从头开始 |
| `ONLINE_EVAL` | `1`；`0` 禁用训练期间自动评估 |
| `EVAL_FREQ` / `SAVE_LAST_FREQ` | `20` / `20` epoch |
| `CHECKPOINT` | 采样的准确 checkpoint 文件路径 |
| `GEN_BSZ` / `NUM_IMAGES` | `256` / `50000` |
| `CLASS_NUM` | 采样默认1000；必须与 checkpoint 一致 |
| `CFG` / `INTERVAL_MIN` / `INTERVAL_MAX` | 采样默认 `2.9` / `0.1` / `1.0` |
| `SAMPLING_METHOD` / `NUM_SAMPLING_STEPS` | 采样默认 `heun` / `50` |
| `EVAL_DIR` | 默认在 checkpoint 同级目录中创建对应采样输出目录 |

单卡、小 batch 的命令检查：

```bash
CUDA_VISIBLE_DEVICES=0 GPUS=1 BATCH_SIZE=2 ONLINE_EVAL=0 \
  bash sh/train_jit_b16_grounding_8gpu.sh --dry-run
```

模型、分辨率、projector 层数/head 数、类别数等结构配置需要与训练 checkpoint 一致。脚本默认 batch 沿用基线设置，新增 projector 的显存与吞吐需通过实际测试确认。

## 显式远端执行

远端已经有相同代码、资源和 SSH 配置时，可显式指定：

```bash
GPU_HOST=my-gpu REMOTE_PROJECT_ROOT=/remote/path/JiT_cond_pixelditv2proj \
  bash sh/train_jit_b16_grounding_8gpu.sh --dry-run
```

去掉 `--dry-run` 才连接远端。运行环境和资源默认在目标机器解析；手动设置的上述环境变量会转发。`run_on_gpu.sh` 使用现有 SSH 配置，不内置集群账号或密钥；可用 `GPU_SSH_KEY` 显式指定私钥路径。它不复制代码、不创建 tmux。

也可以直接使用：

```bash
bash sh/run_on_gpu.sh --dry-run my-gpu bash sh/train_jit_b16_grounding_8gpu.sh
```
