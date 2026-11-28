# 下载 DINOv3-L k=7 统计文件

从 HuggingFace RAEv2 仓库下载统计文件：

```bash
wget https://huggingface.co/nyu-visionx/RAEv2-models/resolve/main/stage1/imagenet/dinov3l-k7/stats.pt \
    -O dinov3l_k7_imagenet1k_stats.pt
```

或使用 huggingface-cli：

```bash
huggingface-cli download nyu-visionx/RAEv2-models \
    stage1/imagenet/dinov3l-k7/stats.pt \
    --local-dir . --local-dir-use-symlinks False
```

文件信息：
- 大小：约 4 KB
- 内容：`{'mean': torch.Tensor([1024]), 'var': torch.Tensor([1024])}`
- 用途：DINOv3-L k=7 latent 的 z-score 归一化
