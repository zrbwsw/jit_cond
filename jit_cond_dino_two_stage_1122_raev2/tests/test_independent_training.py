import copy
from pathlib import Path

import pytest
import torch
from torch import nn

import denoiser
from checkpoint_io import initialize_ema, latent_spec, read_checkpoint, resume_model, use_ema
from main_jit import get_args_parser, validate_args
from util.misc import add_weight_decay, save_model


class TinyEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.embed_dim = 1024
        self.weight = nn.Parameter(torch.tensor(2.0))

    def encode(self, x):
        pooled = torch.nn.functional.adaptive_avg_pool2d(x.mean(1, keepdim=True), 16)
        return pooled.expand(-1, self.embed_dim, -1, -1) * self.weight


class TinyPixel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.2))
        self.calls = []

    def forward(self, z, t, y, grounding_tokens=None, grounding_mask=None, t_dino=None):
        self.calls.append((grounding_tokens, grounding_mask, t_dino))
        condition = grounding_tokens.mean((1, 2))
        if isinstance(grounding_mask, bool) and not grounding_mask:
            condition = torch.zeros_like(condition)
        elif grounding_mask is not None:
            condition = condition * grounding_mask.to(condition.dtype)
        return self.weight * z + condition[:, None, None, None]


@pytest.fixture
def args(tmp_path):
    stats_path = tmp_path / "stats.pt"
    torch.save({
        "mean": torch.zeros(1024, 16, 16),
        "var": torch.ones(1024, 16, 16),
    }, stats_path)
    args = get_args_parser().parse_args(["--dino_stat_path", str(stats_path)])
    args.output_dir = str(tmp_path)
    args.device = "cpu"
    args.class_num = 4
    args.label_drop_prob = 0.0
    args.steps_pixel = 2
    args.latent_spec = latent_spec(args)
    return args


@pytest.fixture(autouse=True)
def tiny_models(monkeypatch):
    monkeypatch.setattr(denoiser, "DINOv3Lk7Encoder", TinyEncoder)
    from model_jit import JiT_models
    monkeypatch.setitem(JiT_models, "JiT-B/16", TinyPixel)


def batch():
    return torch.randn(3, 3, 256, 256), torch.tensor([0, 1, 2])


def test_clean_grounding_and_gradient(args):
    model = denoiser.Denoiser(args).train()
    loss, metrics = model(*batch())
    loss.backward()
    tokens, mask, _ = model.net.calls[0]
    assert tokens.shape == (3, 256, 1024)
    assert mask.all()
    assert metrics["loss_pixel"] == pytest.approx(loss.item())
    assert model.net.weight.grad is not None
    assert model.dino_encoder.weight.grad is None
    initialize_ema(model)
    model.update_ema()
    assert model.method_dino == "raev2"
    assert model.steps_dino == 100
    assert model.cfg_dino == 1.0
    assert model.cfg_interval_dino == (0.0, 1.0)


def test_dropout_masks_grounding(args):
    args.label_drop_prob = 1.0
    model = denoiser.Denoiser(args).train()
    model(*batch())
    _, mask, _ = model.net.calls[0]
    assert not mask.any()


def test_generate_uses_conditional_and_unconditional_branches(args):
    model = denoiser.Denoiser(args).eval()
    model.steps_pixel = 1
    model.sample_dino_latent = lambda labels: torch.ones(
        len(labels), 1024, 16, 16, device=labels.device)
    images = model.generate(torch.tensor([0, 1]))
    assert images.shape == (2, 3, 256, 256)
    assert torch.isfinite(images).all()
    assert len(model.net.calls) == 2
    assert model.net.calls[0][1] is None
    assert model.net.calls[1][1] is False


def test_missing_raev2_sampling_config_fails_clearly(args):
    model = denoiser.Denoiser(args)
    with pytest.raises(ValueError, match="raev2_checkpoint and --raev2_config"):
        model.sample_dino_latent(torch.tensor([0]))
    args.online_eval = True
    with pytest.raises(ValueError, match="raev2_checkpoint and --raev2_config"):
        validate_args(args)


def test_checkpoint_roundtrip(args):
    model = denoiser.Denoiser(args)
    initialize_ema(model)
    optimizer = torch.optim.AdamW(add_weight_decay(model), lr=1e-4)
    save_model(args, model, optimizer, epoch=3, epoch_name="last")
    checkpoint = read_checkpoint(Path(args.output_dir) / "checkpoint-last.pth")
    resumed = denoiser.Denoiser(args)
    resumed_optimizer = torch.optim.AdamW(add_weight_decay(resumed), lr=1e-4)
    resume_model(resumed, resumed_optimizer, checkpoint, args)
    assert args.start_epoch == 3
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key])
    torch.testing.assert_close(model.ema_params1, resumed.ema_params1)


def test_ema_restores_model_on_error(args):
    model = denoiser.Denoiser(args)
    initialize_ema(model)
    original = copy.deepcopy(model.state_dict())
    model.ema_params1[0].fill_(7)
    with pytest.raises(RuntimeError, match="probe"):
        with use_ema(model):
            raise RuntimeError("probe")
    for key, value in original.items():
        torch.testing.assert_close(value, model.state_dict()[key])


def test_official_raev2_sampling_config():
    from omegaconf import OmegaConf
    from configs.stage2 import Stage2Config

    path = "/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/RAEv2/configs/stage2/sampling/imagenet-dinov3l-k7.yaml"
    config = OmegaConf.to_object(OmegaConf.merge(
        OmegaConf.structured(Stage2Config), OmegaConf.load(path)))
    assert config.sampler.num_steps == 100
    assert config.guidance.cfg.scale == 1.0
    assert config.guidance.ig.scale == 1.78
    assert config.guidance.ig.t_min == 0.1
    assert config.guidance.ig.t_max == 1.0
    assert tuple(config.misc.latent_size) == (1024, 16, 16)
