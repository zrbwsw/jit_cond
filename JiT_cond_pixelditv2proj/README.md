# JiT with noisy DINOv3 grounding and the PixelDiT2 projector

This variant starts from `JiT_cond` and uses the **current noisy state** as the grounding input. A frozen DINOv3-B/16 encoder extracts patch tokens; PixelDiT2's trainable, timestep-conditioned projector translates them into JiT's hidden space. Every image patch receives its own AdaLN condition:

```text
z_t = t * x_clean + (1 - t) * noise
raw_tokens = frozen_DINOv3_B(z_t)
grounding_tokens = TimestepProjection(raw_tokens, t, patch_grid)
c_patch = t_embed[:, None, :] + y_embed[:, None, :] + grounding_tokens
```

`x_clean` remains the training target, but it is never passed separately to the grounding encoder. Sampling starts from random noise and class labels; it does not load an original image, a validation reference, or a predicted-clean-image probe.

## Architecture

For JiT-B/16 at 256×256:

```text
z_t:                          [B, 3, 256, 256]
DINOv3-B patch tokens:         [B, 256, 768]
projector input Linear:       [B, 256, 768]
time-conditioned DiT block:   [B, 256, 768]
projector output Linear:      [B, 256, 768]
patch AdaLN condition:        [B, 256, 768]
JiT output:                   [B, 3, 256, 256]
```

The projector defaults to one Transformer block, hidden size equal to JiT's hidden size, and attention head count inherited from JiT (768/12 for B). It has its own timestep embedding, RMSNorm, Q/K normalization, two-dimensional RoPE, SwiGLU, and six-way AdaLN modulation. Its input/output Linear layers perform feature translation and width conversion, so there is no additional old `grounding_proj` Linear layer.

Projector configuration is exposed as `--grounding_proj_blocks` and `--grounding_proj_heads`. Omitting the head argument uses the JiT head count. A checkpoint must be loaded with the same architecture that produced it.

Only the projector block's AdaLN modulation layer is zero-initialized, following PixelDiT2. Its input/output projections retain their default initialization; the entire projector output is not zeroed. Initially the block is an identity residual and becomes timestep-dependent as its modulation learns. JiT keeps its existing zero-initialized residual gates and final prediction layer.

The JiT backbone is retained: image tokens use patch-level AdaLN in both Transformer blocks and the final layer; the 32 in-context tokens use the original global `t+y` condition. No extra PixelDiT2 `SiLU(c)` is inserted before JiT's existing AdaLN SiLU.

## Noisy inputs and time convention

The projector and JiT receive the same original time `t∈[0,1]`: `t=0` is pure noise and `t=1` is the clean endpoint. No `1-t` conversion or factor of 1000 is used.

DINO preprocessing applies `(z_t+1)/2`, ImageNet normalization, and any required spatial resize. Noisy states are **not clamped** to `[-1,1]` or `[0,1]`. To match PixelDiT2, the encoder input grid is aligned before DINO extraction: its spatial size is `JiT patch grid × DINO patch size`. For example, JiT-B/32 at 512px has a 16×16 target grid, so the DINO-B/16 input is resized to 256px and produces 256 tokens.

DINO stays in eval mode with gradients disabled. JiT and the projector are trainable and included in the optimizer and both EMA copies. The checkpoint model state includes the frozen DINO parameters; EMA dictionaries contain only trainable parameters.

## Training and sampling

Each training batch first constructs `z_t`, then encodes that same tensor with DINO. The existing JiT flow loss is retained. Label dropout also disables grounding for those samples, matching the unconditional CFG path. `--ground_drop_prob` optionally adds independent grounding dropout while keeping labels; its default is zero.

At every sampling velocity evaluation, the encoder/projector recomputes grounding from the current `(z,t)`:

- Euler evaluates the current state once.
- Heun evaluates `(z,t)`, predicts an intermediate Euler state, then evaluates that new state with `t_next`.
- The last step uses Euler, as in JiT.

The conditional CFG branch uses noisy grounding. The unconditional branch uses the null label and no grounding, matching training's coupled dropout. This is the same joint guidance choice used by the referenced PixelDiT2 sampler. No grounding token cache is reused across ODE evaluations.

## Local resources and launchers

The launchers default to **GPU1 (`agent-gpu-l2-1`)**. `sh/portable_env.sh` selects the Python environment and resolves resources on the target host without loading the dataset or weights. Every path can be overridden with environment variables, and `GPU_HOST=local` disables SSH forwarding.

On this server the resolved defaults are:

| Resource | Path |
| --- | --- |
| Python | `/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/envs/irepa-py310-cu126-20261015/bin/python` |
| ImageNet (training only) | `/mnt/shared-storage-user/datafactory-share/gushishen/imagenet` |
| DINOv3-B weights | `/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/pretrain_models/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` |
| DINO source | `../Ours_BaseT_rmsnorm2_Residual_1016/SSL_code/dinov3` |
| FID statistics and Inception weights | `../fid_stats` |

The default training experiment is `JiT-B16-DINOv3B-Noisy-PixelDiT2Proj-B80`: 80 epochs, eight GPUs, per-GPU batch 128, global batch 1024, and base LR `5e-5` (absolute LR `2e-4`). These are inherited training settings, not a measured memory/performance guarantee for the new projector. Online evaluation runs after epochs 20, 40, 60, and 80 (log indices 19, 39, 59, and 79).

```bash
cd /mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/JiT_cond_pixelditv2proj
tmux new -s 'JiT-B16-DINOv3B-Noisy-PixelDiT2Proj-B80'
bash sh/train_jit_b16_grounding_8gpu.sh --dry-run
bash sh/train_jit_b16_grounding_8gpu.sh
```

A smaller local configuration can be supplied without editing the script:

```bash
GPU_HOST=local CUDA_VISIBLE_DEVICES=0 GPUS=1 BATCH_SIZE=2 ONLINE_EVAL=0 \
  bash sh/train_jit_b16_grounding_8gpu.sh --dry-run
```

Sampling only needs a compatible trained checkpoint, local DINO assets, and the FID assets when computing metrics. The launcher passes no dataset path:

```bash
CHECKPOINT=/path/to/new-variant/checkpoint-last.pth \
  bash sh/sample_jit_b16_grounding_8gpu.sh --dry-run
CHECKPOINT=/path/to/new-variant/checkpoint-last.pth \
  bash sh/sample_jit_b16_grounding_8gpu.sh
```

`--dry-run` prints the command and exits without starting workers or creating output directories. See [sh/README.md](sh/README.md) for overrides and explicit SSH execution. Old clean-grounding checkpoints have a different projector structure and cannot be resumed directly; use a new output directory and a checkpoint trained with this variant.

## Validation

The unit tests use small models and mocked DINO tokens to check the noisy-state path, sampling updates, CFG consistency, projector gradients, checkpoint handling, and launcher configuration without a dataset or DINO checkpoint:

```bash
PYTHONDONTWRITEBYTECODE=1 TORCHDYNAMO_DISABLE=1 OMP_NUM_THREADS=2 \
  /mnt/shared-storage-user/datafactory-share/gushishen/agent-z/envs/irepa-py310-cu126-20261015/bin/python -m unittest discover -s tests -v
```

A small real-DINO smoke run uses synthetic training targets and noise-only sampling; it requires the local DINO assets but no ImageNet data:

```bash
source sh/portable_env.sh
TORCHDYNAMO_DISABLE=1 OMP_NUM_THREADS=2 \
  "$PYTHON_BIN" tools/smoke_train.py --tiny --device cuda \
  --grounding_repo "$DINOV3_REPO" --grounding_weights "$DINO_WEIGHTS_PATH"
```

`--tiny` selects a small JiT for a quick integration check. Omit it to exercise JiT-B/16, and use `--smoke_steps` to select at least three optimizer steps. This smoke check does not measure full-training convergence or default-batch memory consumption.

A real DINOv3-B + JiT-B/16 smoke run at 256px, batch size 1, completed four optimizer updates and noise-only Heun sampling (two steps, three DINO evaluations). Projector gradients became nonzero from the third update and 20 projector parameter tensors changed; the frozen encoder received no gradients. Full training and default-batch throughput/memory have not been run.

The projector is adapted from PixelDiT2. Its attribution and terms are preserved under [third_party/pixeldit2](third_party/pixeldit2).
