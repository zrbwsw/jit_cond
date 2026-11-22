# Two-stage DinoDiT and Pixel JiT

Stage 1 trains DinoDiT in the normalized DINO patch space. Stage 2 loads the chosen DinoDiT checkpoint, freezes DinoDiT and the image encoder, and trains Pixel JiT. The entry points are `stage1/main_dino.py` and `main_jit.py`. They share data and optimizer utilities. See [中文改动说明](中文改动说明.md) for the file layout and the three time variables.

## Train

Run on the target GPU node from this directory. The launchers use `torchrun` on that node with eight visible GPUs by default. `PYTHON_BIN` and `CUDA_VISIBLE_DEVICES` override the interpreter and GPU list. The inherited defaults are batch 128 per GPU, base LR `5e-5`, and five warmup epochs.

```bash
bash sh/train_dino.sh --epochs 80
bash sh/train_dino.sh --epochs 160

bash sh/train_pixel.sh --epochs 200 --condition_mode_pixel pred \
  --dino_checkpoint checkpoints/stage1-dinov3b-dinodit-b-e80/checkpoint-80.pth

bash sh/train_pixel.sh --epochs 200 --condition_mode_pixel noised \
  --dino_checkpoint checkpoints/stage1-dinov3b-dinodit-b-e80/checkpoint-80.pth
```

Stage 1 defaults to 80 epochs, Stage 2 to 200 epochs with `pred`. The numbered checkpoint interval defaults to 20. For Stage 2, the name uses the **loaded checkpoint's completed epochs**, even when the parent directory was named for a longer Stage 1 run. The first `pred` example writes to `checkpoints/stage1-dinov3b-dinodit-b-e80-stage2-pixeljit-b-dinodit-pred-e200`. `--output_dir` overrides the generated name. `--dry-run` prints the launch command.

| `--condition_mode_pixel` | Pixel training grounding | Pixel DINO time tag |
| --- | --- | --- |
| `pred` | Frozen DinoDiT single-step prediction of the clean latent, given a noised latent and the class | `1` |
| `noised` | Directly inject the lightly noised clean-image DINO latent | `t_d_p` used for corruption |

Both modes use the same inference path. DinoDiT first samples a latent from noise, and Pixel JiT samples an image using that latent with the DINO time tag `1`. Stage 2 accepts a Stage 1 checkpoint in both modes so that the same frozen teacher is available at inference. The default teacher weights are `--dino_weights ema1`. Use `--resume` with a same-stage checkpoint to restore optimizer, EMA and epoch.

## Sample

```bash
bash sh/sample_pixel.sh \
  --checkpoint checkpoints/stage1-dinov3b-dinodit-b-e80-stage2-pixeljit-b-dinodit-pred-e200/checkpoint-200.pth \
  --cfg_dino 2.0 --cfg_pixel 2.9
```

The sampling scripts support separate `--steps_dino` and `--steps_pixel`, CFG intervals (`--interval_min_dino` / `--interval_max_dino`, `--interval_min_pixel` / `--interval_max_pixel`), and `--noise_scale_dino` / `--noise_scale_pixel`. The default sampler uses Heun with 25 latent and 50 pixel steps. A Stage 2 checkpoint stores the frozen DinoDiT weights alongside the Pixel model. Use `--online_eval` on the Pixel launcher to enable image evaluation during training.

## Data and model resources

The defaults use DINOv3-B/16 and ImageNet 256. Set `--dino_encoder dinov2b` for DINOv2-reg-B/14 in both stages. Both stages need matching DINO encoder weights and latent statistics. For the statistics, use `--dino_stat_path` or place them in `stat/`. The default search also checks the existing `../JiT_cond_join_dino/stat/` and `../jit_cond_predict_independent_dino/stat/` directories. The default filenames are `dinov3b_rae_imagenet1k_stats.pt` and `dinov2b_rae_imagenet1k_stat.pt`.

Use `--data_path` or `IMAGENET_ROOT` for the ImageNet directory, `PRETRAIN_MODELS_ROOT` for pretrained encoder weights, `--dinov3_repo` or `DINOV3_REPO` for the DINOv3 source, and `FID_STATS_ROOT` for image evaluation. Stage 2 checks the latent statistics fingerprint when it is present in the source checkpoint. Compatible checkpoints from the earlier independent and JoinDino variants use the same named weight branches.
