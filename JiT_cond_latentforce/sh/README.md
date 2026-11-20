# JiT_cond 启动脚本

脚本在开发机前台运行，通过 `run_on_gpu.sh` 把同一命令交给 GPU 节点；不会在 GPU 节点再创建 tmux。开发机上先创建实验名 tmux：

```bash
tmux new -s 'JiT-B16-DINOv3B-PatchAdaLN-B80'
bash sh/train_jit_b16_grounding_8gpu.sh --dry-run
bash sh/train_jit_b16_grounding_8gpu.sh
```

训练默认是 JiT-B/16、8 卡、每卡 batch 128（全局 batch 1024）、80 epoch，绝对学习率由 `blr * global_batch / 256` 自动缩放。默认输出和日志在 `checkpoints/JiT-B16-DINOv3B-PatchAdaLN-B80/`。

采样前在脚本顶部修改 `CHECKPOINT`：

```bash
bash sh/sample_jit_b16_grounding_8gpu.sh --dry-run
bash sh/sample_jit_b16_grounding_8gpu.sh
```

`--dry-run` 只打印最终命令。GPU 节点、卡号、Python、数据目录和 DINO 配置都在脚本顶部直接修改。
