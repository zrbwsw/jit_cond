# JiT-B clean-image grounding ablation

This directory starts from JiT-B and adds a frozen DINOv3-B/16 clean-image
grounding branch. The original AdaLN condition is the global vector `t + y`.
With `--use_grounding`, every JiT image patch receives

```text
c[b, patch] = t_embed[b] + y_embed[b] + W_g dino_patch[b, patch]
```

The block and final-layer AdaLN modulation is evaluated per patch. JiT-B/16
and DINOv3-B/16 both produce a 16x16 patch grid at 256px, so tokens align
without interpolation. The implementation also resizes square grids for the
B/32 variant.

During training, DINOv3 sees the clean target image before the flow noise is
added. DINOv3 is frozen and its tokens are detached; only `W_g` and JiT are
optimized.

### Sampling-time semantic source

`--grounding_source` decides where the semantic tokens come from at sampling:

| value | meaning |
| --- | --- |
| `val` (default) | Real validation image of the same class. Generation index `g` uses label `g // 50` and that class's validation image `g % 50`, so every sample is paired with a real reference image. This measures the **upper bound** of patch-level AdaLN when clean semantics are always available. |
| `probe` | Estimate `x_hat` from the `t+y` path, then encode it. Fallback when no reference image exists; needs `--ground_drop_prob` to train that path. |
| `none` | Plain `t+y` sampling, for the ablation baseline. |

`W_g` is zero-initialized, so a freshly constructed grounded model is
numerically identical to the baseline; the semantic branch grows in during
training instead of drowning the `t+y` signal from step 0.

### Masking

Label-dropped samples (the CFG unconditional branch) also have their semantic
condition zeroed. Without this, the DINO tokens of a clean image leak the class
into the unconditional branch and CFG never learns a real null condition.
`--ground_drop_prob` additionally disables semantics for some label-kept
samples, which is what trains the `probe` path.

The local assets are already available at:

```text
ImageNet:       /mnt/shared-storage-user/datafactory-share/gushishen/imagenet
DINOv3-B/16:    /mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/pretrain_models/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
FID stats:      /mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/fid_stats
DINO source:    ../Ours_BaseT_rmsnorm2_Residual_1016/SSL_code/dinov3
```

No native JiT-B checkpoint is present in `pretrain_models`; the grounding
experiment therefore starts from scratch unless `--resume` or `CHECKPOINT` is
set to a compatible JiT-cond checkpoint.

## Default 8-GPU experiment

The shared `irepa-py310-cu126-20261015` Python environment is selected by
default. The training launcher uses JiT-B/16 at 256px for 80 epochs, eight
GPUs, and a global batch size of 1024 (`--batch_size 128` on each GPU). The
main program computes the absolute learning rate as
`blr * global_batch_size / 256`, so the default `blr=5e-5` becomes `2e-4`.

Create a development-machine tmux session named after the experiment, then run
the foreground launcher from `JiT_cond`:

```bash
tmux new -s 'JiT-B16-DINOv3B-PatchAdaLN-B80'
bash sh/train_jit_b16_grounding_8gpu.sh --dry-run
bash sh/train_jit_b16_grounding_8gpu.sh
```

The script forwards the same command through `sh/run_on_gpu.sh`; it does not
create another tmux session on the GPU host. Output and `train.log` default to
`JiT_cond/checkpoints/JiT-B16-DINOv3B-PatchAdaLN-B80`.

Sampling is configured at the top of `sh/sample_jit_b16_grounding_8gpu.sh`.
Use a checkpoint file so epoch, sampler, CFG, and interval are visible in the
evaluation directory:

先在脚本顶部修改 `CHECKPOINT`，再运行：

```bash
bash sh/sample_jit_b16_grounding_8gpu.sh --dry-run
bash sh/sample_jit_b16_grounding_8gpu.sh
```

GPU 节点、卡号、数据和权重路径都在脚本顶部直接配置；`sh/` 是本项目唯一的实验启动脚本目录。
