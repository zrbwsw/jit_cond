"""Exercise the JiT wrapper with synthetic features and no external checkpoints."""
import pytest
import torch
from torch import nn

import checkpoint_io
import denoiser
from main_jit import get_args_parser
from pixeldit2_c2i import PixelDiT2
from repa import build_repa_projection


class SyntheticEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.last_input = None

    def encode(self, clean):
        self.last_input = clean.detach()
        grid = clean[:, :1, ::16, ::16]
        return grid.expand(-1, 1024, -1, -1) * self.weight

    def forward(self, clean, patch_size):
        self.last_input = clean.detach()
        return clean[:, :1, ::patch_size, ::patch_size].flatten(2).transpose(1, 2).expand(-1, -1, 8)


@pytest.fixture
def pipeline(monkeypatch, tmp_path):
    monkeypatch.setattr(denoiser, 'DINOv3Lk7Encoder', SyntheticEncoder)
    monkeypatch.setattr(denoiser, 'REPATeacher', SyntheticEncoder)
    monkeypatch.setattr(denoiser, 'PixelDiT2', lambda **kw: PixelDiT2(
        hidden_size=16, num_heads=2, depth=2, in_context_start=1, **kw))
    monkeypatch.setattr(denoiser, 'build_repa_projection', lambda: build_repa_projection(16, 8))
    args = get_args_parser().parse_args([])
    args.repa_align_layer = 1
    args.dino_stat_path = str(tmp_path / 'test-stats')
    (tmp_path / 'test-stats').write_bytes(b'test fingerprint')
    args.label_drop_prob = 1.0
    args.steps_pixel = 3
    return args


def test_trainable_params_loss_and_resume(pipeline, tmp_path):
    args = pipeline
    model = denoiser.Denoiser(args).train()
    checkpoint_io.initialize_ema(model)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad))
    images = torch.linspace(-1, 1, 3 * 256 * 256).reshape(1, 3, 256, 256)
    calls = []
    hook = model.net.register_forward_pre_hook(
        lambda module, inputs, kwargs: calls.append((inputs[2], kwargs['grounding_tokens'])),
        with_kwargs=True)
    try:
        loss, metrics = model(images, torch.tensor([13]))
    finally:
        hook.remove()
    clean = (images + 1) / 2
    torch.testing.assert_close(model.dino_encoder.last_input, clean)
    torch.testing.assert_close(model.repa_encoder.last_input, clean)
    labels, tokens = calls[0]
    assert labels.tolist() == [model.num_classes]
    torch.testing.assert_close(tokens[0, :, 0], clean[0, 0, ::16, ::16].reshape(-1))
    assert loss.item() == pytest.approx(metrics['fm_loss'] + 0.5 * metrics['cos_loss'])
    loss.backward()
    assert model.net.grounding_proj.weight.grad is not None
    assert model.repa_projection[0].weight.grad is not None
    for frozen in (model.dino_encoder, model.repa_encoder):
        assert not frozen.training and not frozen.weight.requires_grad and frozen.weight.grad is None
    assert model.raev2_sampler is None
    opt.step()
    model.update_ema()
    saved = checkpoint_io.build_checkpoint(args, model, opt, epoch=7)
    assert set(saved['model_ema']) == {name for name, p in model.named_parameters() if p.requires_grad}
    path = tmp_path / 'pixel.pth'
    torch.save(saved, path)
    restored = denoiser.Denoiser(args)
    restored_opt = torch.optim.AdamW((p for p in restored.parameters() if p.requires_grad))
    raw = checkpoint_io.read_checkpoint(path)
    raw["args"].pop("condition_mode_pixel")
    raw["args"].pop("noise_beta_dino_pixel")
    checkpoint_io.resume_model(restored, restored_opt, raw, args)
    assert args.start_epoch == 7
    assert restored_opt.state_dict()['state'].keys() == opt.state_dict()['state'].keys()
    for name, param in model.named_parameters():
        torch.testing.assert_close(param, restored.get_parameter(name))
    with checkpoint_io.use_ema(restored):
        for name, ema in model.ema_params.items():
            torch.testing.assert_close(restored.get_parameter(name), ema)
    (tmp_path / 'test-stats').write_bytes(b'another latent space')
    with pytest.raises(ValueError, match='statistic'):
        checkpoint_io.resume_model(restored, restored_opt, raw, args)


def test_heun_reuses_one_semantic_sample(pipeline):
    args = pipeline
    args.evaluate_gen = True
    model = denoiser.Denoiser(args).eval()
    assert model.dino_encoder is model.repa_encoder is None
    labels = torch.tensor([13])

    class Sampler:
        def __init__(self):
            self.calls = 0

        def sample_dino_latent(self, batch_labels):
            torch.testing.assert_close(batch_labels, labels)
            self.calls += 1
            return torch.ones(1, 1024, 16, 16)

    sampler = Sampler()
    model.raev2_sampler = sampler
    calls = []
    hook = model.net.register_forward_pre_hook(
        lambda m, inputs, kwargs: calls.append((inputs[1].item(), inputs[2].item(), kwargs)),
        with_kwargs=True)
    try:
        rgb = model.generate(labels)
    finally:
        hook.remove()
    assert sampler.calls == 1
    assert rgb.shape == (1, 3, 256, 256) and rgb.isfinite().all()
    cond, uncond = calls[::2], calls[1::2]
    # Heun predictor + corrector for steps 1 and 2, then a single Euler evaluation.
    assert [t for t, _, _ in cond] == pytest.approx([0, 1/3, 1/3, 2/3, 2/3])
    grounding = cond[0][2]['grounding_tokens']
    assert all(label == 13 and kw['grounding_tokens'] is grounding for _, label, kw in cond)
    assert all(label == model.num_classes and kw['skip_grounding'] for _, label, kw in uncond)


def test_pixel_cfg_interval_and_velocity_scale(pipeline):
    args = pipeline
    args.evaluate_gen = True
    model = denoiser.Denoiser(args)

    class BranchPredictions(nn.Module):
        def forward(self, x, t, y, grounding_tokens=None, skip_grounding=False):
            return torch.zeros_like(x) if skip_grounding else torch.ones_like(x)

    model.net = BranchPredictions()
    times = torch.tensor([0, 0.125, 0.5, 0.9, 0.99]).view(-1, 1, 1, 1)
    labels = torch.zeros(len(times), dtype=torch.long)
    v = model._velocity(torch.zeros_like(times), times, labels, labels + 1000, None)
    expected = torch.tensor([1, 1, 2.4, 1, 1]).view_as(times) / (1 - times).clamp_min(0.05)
    torch.testing.assert_close(v, expected)


def test_release_configuration():
    args = get_args_parser().parse_args([])
    assert (args.model, args.img_size, args.epochs) == ('PixelDiT2-H/16', 256, 160)
    assert (args.condition_mode_pixel, args.noise_beta_dino_pixel) == ('clean', 0.25)
    assert (args.label_drop_prob, args.attn_dropout, args.proj_dropout) == (0.1, 0.0, 0.2)
    assert (args.repa_align_layer, args.repa_weight) == (8, 0.5)
    assert (args.t_mean_pixel, args.t_std_pixel, args.noise_scale_pixel, args.t_eps) == (-0.8, 0.8, 1.0, 0.05)
    assert (args.steps_pixel, args.cfg_pixel, args.interval_min_pixel, args.interval_max_pixel) == (50, 2.4, 0.125, 0.9)
    assert args.lr == 2e-4 and args.warmup_steps == 6255 and args.ema_decay == 0.9999
    assert (args.batch_size, args.accum_steps) == (128, 1)


def test_noised_grounding_matches_time_interpolation(pipeline):
    args = pipeline
    args.condition_mode_pixel = 'noised'
    model = denoiser.Denoiser(args).train()
    images = torch.linspace(-1, 1, 3 * 256 * 256).reshape(1, 3, 256, 256)
    clean = (images + 1) / 2
    clean_latent = model.dino_encoder.encode(clean)
    torch.manual_seed(123)
    tokens, t_dino, t_d_p = model.prepare_grounding_pixel(clean, torch.tensor([13]))
    torch.manual_seed(123)
    expected_t = 0.75 + 0.25 * torch.rand(1)
    expected = expected_t.view(-1, 1, 1, 1) * clean_latent
    expected = expected + (1 - expected_t.view(-1, 1, 1, 1)) * torch.randn_like(clean_latent)
    torch.testing.assert_close(t_dino, expected_t)
    torch.testing.assert_close(t_d_p, expected_t)
    torch.testing.assert_close(tokens, denoiser.latent_to_tokens(expected))
    loss, metrics = model(images, torch.tensor([13]))
    assert 0.75 <= metrics['t_d_p'] < 1
    loss.backward()
    assert model.net.t_dino_embedder.mlp[0].weight.grad is not None


@pytest.mark.parametrize('mode', ['noised', 'pred'])
def test_noised_generation_uses_clean_time_tag(pipeline, mode):
    args = pipeline
    args.condition_mode_pixel = mode
    args.evaluate_gen = True
    model = denoiser.Denoiser(args).eval()
    model.steps_pixel = 1
    model.raev2_sampler = type('Sampler', (), {
        'sample_dino_latent': lambda self, labels: torch.ones(len(labels), 1024, 16, 16),
    })()
    labels = torch.tensor([13])
    calls = []
    hook = model.net.register_forward_pre_hook(
        lambda m, inputs, kwargs: calls.append(kwargs), with_kwargs=True)
    try:
        model.generate(labels)
    finally:
        hook.remove()
    assert len(calls) == 2
    torch.testing.assert_close(calls[0]['t_dino'], torch.ones(1))
    assert 't_dino' not in calls[1] and calls[1]['skip_grounding']


def test_pred_grounding_uses_teacher_prediction_and_clean_tag(pipeline):
    args = pipeline
    args.condition_mode_pixel = 'pred'
    model = denoiser.Denoiser(args).train()
    images = torch.linspace(-1, 1, 3 * 256 * 256).reshape(1, 3, 256, 256)
    clean = (images + 1) / 2
    clean_latent = model.dino_encoder.encode(clean)
    labels = torch.tensor([13])

    class Predictor:
        def predict_clean(self, z_cond, t_dino, batch_labels):
            torch.testing.assert_close(batch_labels, labels)
            self.t_dino = t_dino
            return z_cond * 0.5 + 0.25

    model.raev2_sampler = Predictor()
    torch.manual_seed(123)
    tokens, t_tag, t_d_p = model.prepare_grounding_pixel(clean, labels)
    torch.manual_seed(123)
    expected_t = 0.75 + 0.25 * torch.rand(1)
    expected = expected_t.view(-1, 1, 1, 1) * clean_latent
    expected = expected + (1 - expected_t.view(-1, 1, 1, 1)) * torch.randn_like(clean_latent)
    expected = expected * 0.5 + 0.25
    torch.testing.assert_close(tokens, denoiser.latent_to_tokens(expected))
    torch.testing.assert_close(t_tag, torch.ones_like(expected_t))
    torch.testing.assert_close(t_d_p, expected_t)
    torch.testing.assert_close(model.raev2_sampler.t_dino, expected_t)


def test_raev2_predict_clean_uses_complementary_time_and_full_output():
    from raev2_latent_sampler import RAEv2LatentSampler

    sampler = RAEv2LatentSampler.__new__(RAEv2LatentSampler)
    sampler.device = torch.device("cpu")
    sampler.latent_size = (1024, 16, 16)

    class Model(nn.Module):
        def forward(self, x, t, **kwargs):
            self.t = t
            self.kwargs = kwargs
            return x + 1, x - 1

    sampler.model = Model()
    z_cond = torch.randn(2, 1024, 16, 16)
    t_dino = torch.tensor([0.75, 0.9])
    labels = torch.tensor([3, 4])
    prediction = sampler.predict_clean(z_cond, t_dino, labels)
    torch.testing.assert_close(prediction, z_cond + 1)
    torch.testing.assert_close(sampler.model.t, 1 - t_dino)
    torch.testing.assert_close(sampler.model.kwargs["context"], labels)
    assert sampler.model.kwargs["attn_mask"] is None
