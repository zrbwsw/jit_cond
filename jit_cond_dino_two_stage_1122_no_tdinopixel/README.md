# Two-stage DinoDiT and Pixel JiT (_no_tdinopixel)

Stage 1 trains DinoDiT in the normalized DINO patch space. Stage 2 loads the chosen DinoDiT checkpoint, freezes DinoDiT and the image encoder, and trains Pixel JiT. The entry points are `stage1/main_dino.py` and `main_jit.py`. They share data and optimizer utilities. See [中文改动说明](中文改动说明.md) for the file layout and the three time variables.

## Train

Run on the target GPU node from this directory. The launchers use `torchrun` on that node with eight visible GPUs by default. `PYTHON_BIN` and `CUDA_VISIBLE_DEVICES` override the interpreter and GPU list. The inherited defaults are batch 128 per GPU, base LR `5e-5`, and five warmup epochs.

```bash
bash sh/train_dino.sh --epochs 80
bash sh/train_dino.sh --epochs 160

bash sh/train_pixel.sh --epochs 200 --condition_mode_pixel pred \
  --dino_checkpoint ../jit_cond_dino_two_stage_1122/checkpoints/stage1-dinov3b-dinodit-b-e80/checkpoint-80.pth

bash sh/train_pixel.sh --epochs 200 --condition_mode_pixel noised \
  --dino_checkpoint ../jit_cond_dino_two_stage_1122/checkpoints/stage1-dinov3b-dinodit-b-e80/checkpoint-80.pth
```

Stage 1 defaults to 80 epochs, Stage 2 to 200 epochs with `pred`. The numbered checkpoint interval defaults to 20. For Stage 2, the name uses the **loaded checkpoint's completed epochs**, even when the parent directory was named for a longer Stage 1 run. The first `pred` example writes to `checkpoints/stage1-dinov3b-dinodit-b-e80-stage2-pixeljit-b-dinodit-pred-e200`. `--output_dir` overrides the generated name. `--dry-run` prints the launch command.

| `--condition_mode_pixel` | Training grounding |
| --- | --- |
| `clean` | Normalized patch tokens of the clean image |
| `noised` | Clean latent interpolated with noise using the sampled corruption time |
| `pred` | Frozen DinoDiT single-step prediction from that noised latent, corruption time and class |

All three modes use `E_pixel(t_pixel) + E_y(y) + W_g(z_cond)` for grounded Pixel patches. At inference, DinoDiT samples the grounding latent once per batch. Its integration times stay within DinoDiT. The default teacher weights are `--dino_weights ema1`. Use `--resume` with a same-stage checkpoint to restore optimizer, EMA and epoch.

## Sample

```bash
bash sh/sample_pixel.sh \
  --checkpoint checkpoints/stage1-dinov3b-dinodit-b-e80-stage2-pixeljit-b-dinodit-pred-e200/checkpoint-200.pth \
  --cfg_dino 2.0 --cfg_pixel 2.9
```

The sampling scripts support separate `--steps_dino` and `--steps_pixel`, CFG intervals (`--interval_min_dino` / `--interval_max_dino`, `--interval_min_pixel` / `--interval_max_pixel`), and `--noise_scale_dino` / `--noise_scale_pixel`. The default sampler uses Heun with 25 latent and 50 pixel steps. A Stage 2 checkpoint stores the frozen DinoDiT weights alongside the Pixel model. Use `--online_eval` on the Pixel launcher to enable image evaluation during training.

## Real-val grounding (noised / pred)

`noised_grounding_evaluation.py` evaluates Stage 2 checkpoints with training-matched validation-image conditions. It encodes each real validation image and constructs a noised latent with random `t_dino_pixel ~ U[0.75,1)` or an optional fixed value. For `noised`, Pixel receives the noised tokens. For `pred`, DinoDiT uses the corruption time to predict the tokens supplied to Pixel.

```bash
bash sh/sample_stage2_realval_noised_grounding.sh --dry-run
GPU_HOST=agent-gpu-l2-1 bash sh/sample_stage2_realval_noised_grounding.sh

# Fixed 25% noise: z_cond = 0.75 * clean + 0.25 * noise
FIXED_T_DINO_PIXEL=0.75 GPU_HOST=agent-gpu-l2-1 \
  bash sh/sample_stage2_realval_noised_grounding.sh

# Pred Stage 2 checkpoint
CONDITION_MODE=pred CHECKPOINT=/path/to/pred/checkpoint-160.pth \
  GPU_HOST=agent-gpu-l2-1 bash sh/sample_stage2_realval_noised_grounding.sh

# Keep generated PNG files after FID/IS evaluation
KEEP_IMAGES=1 GPU_HOST=agent-gpu-l2-1 \
  bash sh/sample_stage2_realval_noised_grounding.sh
```

The script defaults to the E160 noised checkpoint, 50K images, Pixel Heun-50 and CFG 2.9. Set `CHECKPOINT`, `CONDITION_MODE`, `JIT_CFG`, `EVAL_DIR` or `GPU_HOST` to override these defaults. `FIXED_T_DINO_PIXEL` overrides random time sampling and changes the default evaluation directory to include `realval_<mode>_<value>`. `KEEP_IMAGES=1` preserves generated images; the default still removes them after metrics.

## Data and model resources

The defaults use DINOv3-B/16 and ImageNet 256. Set `--dino_encoder dinov2b` for DINOv2-reg-B/14 in both stages. Both stages need matching DINO encoder weights and latent statistics. For the statistics, use `--dino_stat_path` or place them in `stat/`. The default search also checks the existing `../JiT_cond_join_dino/stat/` and `../jit_cond_predict_independent_dino/stat/` directories. The default filenames are `dinov3b_rae_imagenet1k_stats.pt` and `dinov2b_rae_imagenet1k_stat.pt`.

Use `--data_path` or `IMAGENET_ROOT` for the ImageNet directory, `PRETRAIN_MODELS_ROOT` for pretrained encoder weights, `--dinov3_repo` or `DINOV3_REPO` for the DINOv3 source, and `FID_STATS_ROOT` for image evaluation. Stage 2 checks the latent statistics fingerprint when it is present in the source checkpoint. Compatible checkpoints from the earlier independent and JoinDino variants use the same named weight branches.
