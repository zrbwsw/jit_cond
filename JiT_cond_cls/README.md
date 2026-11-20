# JiT-B clean-image CLS grounding ablation

This directory starts from JiT-B and adds a frozen DINOv3-B/16 clean-image
CLS grounding branch. The original AdaLN condition is the global vector `t + y`.
With `--use_grounding`, the condition becomes

```text
c[b] = t_embed[b] + y_embed[b] + W_g dino_cls[b]
```

Each Transformer block uses its own AdaLN projection to generate shift, scale,
and residual gates from this vector. All tokens within a sample share the same
condition, so this is **block-level (global) AdaLN**, unlike the patch-level
variant in `JiT_cond`. The final layer uses the same global condition. Only the
normalized DINOv3 CLS token is used; patch and register tokens are discarded.

During training, DINOv3 sees the clean target image before flow noise is added.
DINOv3 is frozen and its tokens are detached; only `W_g` and JiT are optimized.
`W_g` is zero-initialized, so a freshly constructed grounded model is
numerically identical to the baseline.

### Sampling-time semantic source

`--grounding_source` decides where the semantic token comes from at sampling:

| value | meaning |
| --- | --- |
| `val` (default) | Real validation image of the same class. Generation index `g` uses label `g // 50` and that class's validation image `g % 50`. This measures the upper bound of CLS block-level AdaLN when clean semantics are available. |
| `probe` | Estimate `x_hat` from the `t+y` path, then encode its CLS token. This fallback needs `--ground_drop_prob` to train the no-grounding path. |
| `none` | Plain `t+y` sampling, for the ablation baseline. |

### Masking

Label-dropped samples (the CFG unconditional branch) also have their semantic
condition zeroed, preventing clean-image class leakage into the unconditional
branch. `--ground_drop_prob` additionally disables semantics for label-kept
samples, which trains the `probe` path.

The local assets are already available at:

```text
ImageNet:       /mnt/shared-storage-user/datafactory-share/gushishen/imagenet
DINOv3-B/16:    /mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/pretrain_models/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
FID stats:      /mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/fid_stats
DINO source:    ../Ours_BaseT_rmsnorm2_Residual_1016/SSL_code/dinov3
```

No native JiT-B checkpoint is required. The experiment starts from scratch
unless `--resume` or `CHECKPOINT` points to a compatible `jit_cond_cls` checkpoint.

## Default 8-GPU experiment

The training launcher uses JiT-B/16 at 256px for 600 epochs on GPU2
(`agent-gpu-l2-2`), with eight GPUs and global batch size 1024
(`--batch_size 128` per GPU). The absolute learning rate is
`blr * global_batch_size / 256`, so `blr=5e-5` becomes `2e-4`.

Create a development-machine tmux session named after the experiment, then run
the foreground launcher from `jit_cond_cls`:

```bash
tmux new -s 'JiT-B16-DINOv3B-CLS-BlockAdaLN-ValGrounding-B600'
bash sh/train_jit_b16_grounding_8gpu.sh --dry-run
bash sh/train_jit_b16_grounding_8gpu.sh
```

The script forwards the command through `sh/run_on_gpu.sh`; it does not create
another tmux session on the GPU host. Output and `train.log` default to
`jit_cond_cls/checkpoints/JiT-B16-DINOv3B-CLS-BlockAdaLN-ValGrounding-B600`.

Sampling is configured at the top of `sh/sample_jit_b16_grounding_8gpu.sh`.
Set `CHECKPOINT` to the desired file, then run:

```bash
bash sh/sample_jit_b16_grounding_8gpu.sh --dry-run
bash sh/sample_jit_b16_grounding_8gpu.sh
```
