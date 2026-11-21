import copy
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import denoiser
import denoiser_dino
from checkpoint_io import initialize_ema, latent_spec, load_branch, read_checkpoint, resume_model, use_ema
from main_jit import get_args_parser, validate_args
from util.misc import add_weight_decay, save_model


class TinyEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))

    def encode(self, x):
        return torch.nn.functional.adaptive_avg_pool2d(x.mean(1, keepdim=True), (2, 2)).expand(-1, 8, -1, -1) * self.weight


class TinyDino(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.3))
        self.pos_embed = nn.Parameter(torch.tensor(0.02), requires_grad=False)
        self.dropout = nn.Dropout(0.9)
        self.calls = []

    def forward(self, z, t, y):
        result = self.dropout(z) * self.weight + self.pos_embed
        self.calls.append((z, t, y, result, torch.is_grad_enabled()))
        return result


class TinyPixel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.2))
        self.pos_embed = nn.Parameter(torch.tensor(0.), requires_grad=False)
        self.calls = []

    def forward(self, z, t, y, grounding_tokens=None, grounding_mask=None, t_dino=None):
        self.calls.append((grounding_tokens, grounding_mask, t_dino))
        condition = grounding_tokens.mean((1, 2))
        if grounding_mask is not None:
            condition = condition * grounding_mask
        return self.weight * z + condition[:, None, None, None]


@pytest.fixture
def args(tmp_path):
    args = get_args_parser().parse_args([])
    args.img_size = 32
    args.grounding_dim = 8
    args.class_num = 4
    args.dino_encoder = 'dinov3b'
    args.output_dir = str(tmp_path)
    args.device = 'cpu'
    args.dino_steps = args.num_sampling_steps = 2
    args.latent_spec = latent_spec(args)
    return args


@pytest.fixture(autouse=True)
def tiny_models(monkeypatch):
    torch.set_num_threads(2)
    monkeypatch.setitem(denoiser.JiT_models, 'JiT-B/16', TinyPixel)
    monkeypatch.setitem(denoiser_dino.DinoDiT_models, 'DinoDiT-B', TinyDino)
    monkeypatch.setattr(denoiser_dino, 'DINOLatentEncoder', TinyEncoder)


def batch():
    return torch.randn(3, 3, 32, 32), torch.tensor([0, 1, 2])


def test_pixel_single_predict_freeze_and_condition_tag(args):
    model = denoiser.Denoiser(args).train()
    assert model.net.training
    assert not model.dino_net.training and not model.dino_encoder.training
    x, labels = batch()
    loss, metrics = model(x, labels)
    loss.backward()
    assert set(metrics) == {'loss_jit', 'condition_time'}
    assert len(model.dino_net.calls) == 1
    _, s, seen_labels, prediction, grad_enabled = model.dino_net.calls[0]
    tokens, _, tag = model.net.calls[0]
    torch.testing.assert_close(tokens, model._as_tokens(prediction))
    torch.testing.assert_close(seen_labels, labels)
    assert torch.all((s >= .75) & (s < 1))
    assert torch.all(tag == 1)
    assert not grad_enabled and not tokens.requires_grad
    assert model.net.weight.grad is not None
    assert all(p.grad is None and not p.requires_grad for p in model.dino_net.parameters())
    assert all(p.grad is None and not p.requires_grad for p in model.dino_encoder.parameters())
    params = [p for group in add_weight_decay(model) for p in group['params']]
    assert {id(p) for p in params} == {id(model.net.weight)}


def test_label_dropout_masks_pixel_not_teacher(args):
    args.label_drop_prob = 1.0
    model = denoiser.Denoiser(args)
    x, labels = batch()
    model(x, labels)
    assert not model.net.calls[0][1].any()
    torch.testing.assert_close(model.dino_net.calls[0][2], labels)


def test_logit_normal_condition_schedule(args, monkeypatch):
    args.condition_time_schedule = 'logit_normal'
    model = denoiser.Denoiser(args)
    def sample(n, device, mean, std):
        assert (mean, std) == (args.D_mean, args.D_std)
        return torch.full((n,), .2, device=device)
    monkeypatch.setattr(model, 'sample_t', sample)
    _, s = model.predict_grounding(*batch())
    torch.testing.assert_close(s, torch.full_like(s, .2))


def test_dino_training_has_no_pixel_and_only_dino_gradients(args):
    model = denoiser_dino.DinoDenoiser(args).train()
    assert not hasattr(model, 'net')
    assert model.dino_net.training and not model.dino_encoder.training
    loss, metrics = model(*batch())
    loss.backward()
    assert set(metrics) == {'loss_dino'}
    assert model.dino_net.weight.grad is not None
    assert model.dino_encoder.weight.grad is None


def legacy_checkpoint(args):
    model = denoiser.Denoiser(args)
    state = copy.deepcopy(model.state_dict())
    state['dino_net.weight'].fill_(.3)
    state['net.weight'].fill_(.2)
    # Old joint EMA only includes trainable weights, not fixed positions/encoder.
    return {
        'args': copy.copy(args), 'model': state,
        'model_ema1': {'dino_net.weight': torch.tensor(.7), 'net.weight': torch.tensor(.6)},
        'model_ema2': {'dino_net.weight': torch.tensor(.5), 'net.weight': torch.tensor(.4)},
        'optimizer': {'invalid_joint_optimizer': True}, 'epoch': 80,
    }


@pytest.mark.parametrize('source,expected', [('model', .3), ('ema1', .7), ('ema2', .5)])
def test_legacy_dino_extraction_named_ema(args, source, expected):
    model = denoiser.Denoiser(args)
    checkpoint = legacy_checkpoint(args)
    pixel_before = model.net.weight.clone()
    load_branch(model, checkpoint, 'dino_net', source, args)
    assert model.dino_net.weight.item() == pytest.approx(expected)
    torch.testing.assert_close(model.net.weight, pixel_before)
    assert not model.dino_net.weight.requires_grad
    assert args.start_epoch == 0


def test_pixel_warm_start_does_not_replace_dino(args):
    model = denoiser.Denoiser(args)
    before = model.dino_net.weight.clone()
    load_branch(model, legacy_checkpoint(args), 'net', 'ema1', args)
    assert model.net.weight.item() == pytest.approx(.6)
    torch.testing.assert_close(before, model.dino_net.weight)


def test_reject_missing_ema_or_wrong_latent(args):
    model = denoiser.Denoiser(args)
    checkpoint = legacy_checkpoint(args)
    del checkpoint['model_ema1']['dino_net.weight']
    with pytest.raises(ValueError, match='missing'):
        load_branch(model, checkpoint, 'dino_net', 'ema1', args)
    checkpoint = legacy_checkpoint(args)
    checkpoint['args'].dino_encoder = 'dinov2b'
    with pytest.raises(ValueError, match='dino_encoder'):
        load_branch(model, checkpoint, 'dino_net', 'model', args)
    checkpoint = legacy_checkpoint(args)
    checkpoint['latent_spec'] = {'stats_sha256': 'different'}
    with pytest.raises(ValueError, match='normalization'):
        load_branch(model, checkpoint, 'dino_net', 'ema1', args)


@pytest.mark.parametrize('stage', ['pixel', 'dino'])
def test_checkpoint_roundtrip_and_stage_guard(args, stage):
    args.training_stage = stage
    cls = denoiser.Denoiser if stage == 'pixel' else denoiser_dino.DinoDenoiser
    model = cls(args)
    initialize_ema(model)
    optimizer = torch.optim.AdamW(add_weight_decay(model), lr=1e-4)
    loss, _ = model(*batch())
    loss.backward()
    optimizer.step()
    model.update_ema()
    save_model(args, model, optimizer, epoch=3, epoch_name='last')
    checkpoint = read_checkpoint(Path(args.output_dir) / 'checkpoint-last.pth')
    prefix = 'net.' if stage == 'pixel' else 'dino_net.'
    assert all(k.startswith(prefix) for k in checkpoint['model_ema1'])
    other = cls(args)
    other_optimizer = torch.optim.AdamW(add_weight_decay(other), lr=1e-4)
    resume_model(other, other_optimizer, checkpoint, args)
    assert args.start_epoch == 3
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, other.state_dict()[key])
    torch.testing.assert_close(model.ema_params1, other.ema_params1)
    assert len(optimizer.state) == len(other_optimizer.state) > 0
    checkpoint['training_stage'] = 'dino' if stage == 'pixel' else 'pixel'
    with pytest.raises(ValueError, match='other-stage'):
        resume_model(other, other_optimizer, checkpoint, args)


def test_joint_optimizer_resume_rejected(args):
    model = denoiser.Denoiser(args)
    optimizer = torch.optim.AdamW(add_weight_decay(model))
    with pytest.raises(ValueError, match='Cannot resume'):
        resume_model(model, optimizer, legacy_checkpoint(args), args)


def test_ema_keeps_teacher_fixed_and_restores_on_error(args):
    model = denoiser.Denoiser(args)
    initialize_ema(model)
    original = model.net.weight.clone()
    dino_original = model.dino_net.weight.clone()
    model.ema_params1[0].fill_(9)
    with pytest.raises(RuntimeError, match='probe'):
        with use_ema(model):
            assert model.net.weight.item() == 9
            torch.testing.assert_close(model.dino_net.weight, dino_original)
            raise RuntimeError('probe')
    torch.testing.assert_close(model.net.weight, original)


@pytest.mark.parametrize('method', ['heun', 'euler'])
def test_generation_is_multistep_and_does_not_use_encoder(args, monkeypatch, method):
    args.sampling_method = args.dino_sampling_method = method
    model = denoiser.Denoiser(args).eval()
    def unexpected(*args):
        raise AssertionError('generation cannot encode a real image')
    monkeypatch.setattr(model.dino_encoder, 'encode', unexpected)
    images = model.generate(torch.tensor([0, 1]))
    assert images.shape == (2, 3, 32, 32)
    assert torch.isfinite(images).all()
    assert len(model.dino_net.calls) == (3 if method == 'heun' else 2)
    assert all(torch.all(tag == 1) for _, _, tag in model.net.calls)


def test_fail_fast_without_frozen_pretraining_or_missing_resume(tmp_path):
    args = get_args_parser().parse_args([])
    with pytest.raises(ValueError, match='requires --dino_checkpoint'):
        validate_args(args)
    args.resume = str(tmp_path / 'missing')
    with pytest.raises(FileNotFoundError, match='Resume'):
        validate_args(args)
    args.resume = ''
    args.training_stage = 'dino'
    assert validate_args(args) is None
    args.online_eval = True
    with pytest.raises(ValueError, match='no Pixel'):
        validate_args(args)


@torch.compiler.disable
def test_actual_transformer_gradient_and_generation(args, monkeypatch):
    from model_jit import JiT
    from latent_force.dino_dit import DinoDiT
    monkeypatch.setitem(denoiser.JiT_models, 'JiT-B/16', lambda **kw: JiT(
        hidden_size=64, depth=2, num_heads=4, bottleneck_dim=8,
        in_context_len=2, in_context_start=1, **kw))
    monkeypatch.setitem(denoiser_dino.DinoDiT_models, 'DinoDiT-B', lambda **kw: DinoDiT(
        hidden_size=64, depth=2, num_heads=4, bottleneck_dim_dino=8,
        in_context_len=2, in_context_start=1, **kw))
    model = denoiser.Denoiser(args).train()
    # Avoid zero-initialized output/projection obscuring the condition gradient test.
    torch.nn.init.normal_(model.net.final_layer.linear.weight, std=.01)
    torch.nn.init.normal_(model.net.final_layer.adaLN_modulation[-1].weight, std=.01)
    torch.nn.init.normal_(model.dino_net.final_layer.linear.weight, std=.01)
    loss, _ = model(*batch())
    loss.backward()
    assert torch.isfinite(loss)
    assert model.net.grounding_proj.weight.grad is not None
    assert model.net.grounding_proj.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.dino_net.parameters())
    images = model.eval().generate(torch.tensor([0]))
    assert images.shape == (1, 3, 32, 32) and torch.isfinite(images).all()


def test_evaluate_restores_pixel_and_never_changes_teacher(args):
    from engine_jit import evaluate
    model = denoiser.Denoiser(args).train()
    initialize_ema(model)
    model.ema_params1[0].fill_(.7)
    before = copy.deepcopy(model.state_dict())
    args.num_images = 4
    evaluate(model, args, 1, batch_size=2, log_writer=None)
    assert len(list(Path(args.output_dir).rglob('*.png'))) == 4
    assert model.training and model.net.training and not model.dino_net.training
    for name, value in before.items():
        torch.testing.assert_close(value, model.state_dict()[name])


def test_end_to_end_dino_then_pixel_and_resume_cpu(args, monkeypatch, tmp_path):
    import main_jit
    # Tiny models use 32px while production validation deliberately requires 256px.
    monkeypatch.setattr(main_jit, 'validate_args', lambda args: args.resume or None)
    images = torch.randint(0, 256, (4, 3, 32, 32), dtype=torch.uint8)
    dataset = torch.utils.data.TensorDataset(images, torch.tensor([0, 1, 2, 3]))
    monkeypatch.setattr(main_jit.datasets, 'ImageFolder', lambda *a, **kw: dataset)
    args.num_workers = 0
    args.pin_mem = False
    args.batch_size = 2
    args.epochs = args.save_last_freq = args.save_interval = 1
    args.warmup_epochs = 0
    args.training_stage = 'dino'
    args.output_dir = str(tmp_path / 'dino')
    main_jit.main(args)
    teacher_path = Path(args.output_dir) / 'checkpoint-last.pth'
    dino_checkpoint = read_checkpoint(teacher_path)
    assert not any(k.startswith('net.') for k in dino_checkpoint['model'])
    args.training_stage = 'pixel'
    args.output_dir = str(tmp_path / 'pixel')
    args.dino_checkpoint = str(teacher_path)
    main_jit.main(args)
    pixel_path = Path(args.output_dir) / 'checkpoint-last.pth'
    first = read_checkpoint(pixel_path)
    torch.testing.assert_close(first['model']['dino_net.weight'], dino_checkpoint['model_ema1']['dino_net.weight'])
    args.resume = str(pixel_path)
    args.dino_checkpoint = ''
    args.epochs = 2
    # Materialize expected values before checkpoint-last is replaced.
    teacher_weight = first['model']['dino_net.weight'].clone()
    del first
    main_jit.main(args)
    resumed = read_checkpoint(pixel_path)
    assert resumed['epoch'] == 2
    torch.testing.assert_close(resumed['model']['dino_net.weight'], teacher_weight)


def test_failed_save_keeps_previous_checkpoint(args, monkeypatch):
    model = denoiser.Denoiser(args)
    initialize_ema(model)
    optimizer = torch.optim.AdamW(add_weight_decay(model))
    save_model(args, model, optimizer, epoch=1, epoch_name='last')
    target = Path(args.output_dir) / 'checkpoint-last.pth'
    before = target.read_bytes()
    def failed_save(*a, **kw):
        raise OSError('simulated disk failure')
    monkeypatch.setattr(torch, 'save', failed_save)
    with pytest.raises(OSError, match='disk failure'):
        save_model(args, model, optimizer, epoch=2, epoch_name='last')
    assert target.read_bytes() == before
    assert not list(Path(args.output_dir).glob('.checkpoint-*.tmp'))


def test_resume_rejects_changed_condition_schedule(args):
    model = denoiser.Denoiser(args)
    initialize_ema(model)
    optimizer = torch.optim.AdamW(add_weight_decay(model))
    save_model(args, model, optimizer, epoch=1, epoch_name='last')
    checkpoint = read_checkpoint(Path(args.output_dir) / 'checkpoint-last.pth')
    args.condition_time_schedule = 'logit_normal'
    with pytest.raises(ValueError, match='condition_time_schedule'):
        resume_model(model, optimizer, checkpoint, args)
