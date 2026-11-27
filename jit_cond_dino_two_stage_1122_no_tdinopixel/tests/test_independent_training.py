import copy
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import denoiser
from stage1 import denoiser_dino, main_dino
from checkpoint_io import LEGACY_KEYS, initialize_ema, latent_spec, load_branch, read_checkpoint, resume_model, use_ema
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

    def forward(self, z, t, y, grounding_tokens=None, grounding_mask=None):
        self.calls.append((grounding_tokens, grounding_mask))
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
    args.steps_dino = args.steps_pixel = 2
    stats = tmp_path / 'stats.pt'
    torch.save({'mean': torch.zeros(8), 'var': torch.ones(8)}, stats)
    args.dino_stat_path = str(stats)
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


@pytest.mark.parametrize('mode', ['clean', 'noised', 'pred'])
def test_pixel_condition_and_frozen_teacher(args, mode):
    args.condition_mode_pixel = mode
    model = denoiser.Denoiser(args).train()
    assert model.net.training
    assert not model.dino_net.training and not model.dino_encoder.training
    x, labels = batch()
    loss, metrics = model(x, labels)
    loss.backward()
    assert set(metrics) == ({'loss_pixel'} if mode == 'clean' else {'loss_pixel', 't_d_p'})
    tokens, _ = model.net.calls[0]
    if mode == 'pred':
        assert len(model.dino_net.calls) == 1
        _, t_d_p, seen_labels, prediction, grad_enabled = model.dino_net.calls[0]
        torch.testing.assert_close(tokens, model._as_tokens(prediction))
        torch.testing.assert_close(seen_labels, labels)
        assert torch.all((t_d_p >= .75) & (t_d_p < 1))
        assert t_d_p.mean().item() == pytest.approx(metrics['t_d_p'])
        assert not grad_enabled
    else:
        assert not model.dino_net.calls
        if mode == "clean":
            torch.testing.assert_close(tokens, model._as_tokens(model.encode_grounding(x)))
        else:
            assert .75 <= metrics["t_d_p"] < 1
    assert not tokens.requires_grad
    assert model.net.weight.grad is not None
    assert all(p.grad is None and not p.requires_grad for p in model.dino_net.parameters())
    assert all(p.grad is None and not p.requires_grad for p in model.dino_encoder.parameters())
    params = [p for group in add_weight_decay(model) for p in group['params']]
    assert {id(p) for p in params} == {id(model.net.weight)}


@pytest.mark.parametrize('mode', ['clean', 'noised', 'pred'])
def test_real_grounding_evaluation_bypasses_dino_and_preserves_noise(args, mode):
    from grounding_evaluation import RealGroundingSampler

    args.condition_mode_pixel = mode

    class Provider:
        def __init__(self):
            self.num_classes = args.class_num
            self.calls = []

        def load_batch(self, labels, intra_indices):
            self.calls.append((labels.copy(), intra_indices.copy()))
            return torch.ones(len(labels), 3, 32, 32)

    provider = Provider()
    model = denoiser.Denoiser(args).eval()
    model.steps_pixel = 1
    model.sample_dino_latent = RealGroundingSampler(model, provider, num_images=8, batch_size=3)
    labels = torch.tensor([0, 0, 1])
    torch.manual_seed(7)
    torch.randn(3, 8, 2, 2)
    torch.randn(3, 3, 32, 32)
    expected_rng_state = torch.random.get_rng_state()
    torch.manual_seed(7)
    images = model.generate(labels)
    assert images.shape == (3, 3, 32, 32)
    assert torch.equal(torch.random.get_rng_state(), expected_rng_state)
    assert provider.calls[0][0].tolist() == [0, 0, 1]
    assert provider.calls[0][1].tolist() == [0, 1, 0]
    assert not model.dino_net.calls
    assert len(model.net.calls) == 2
    assert model.net.calls[-1][1] is False

    rank_sampler = RealGroundingSampler(
        model, provider, False, num_images=8, batch_size=2, world_size=2, rank=1)
    rank_sampler(torch.tensor([1, 1]))
    assert provider.calls[-1][0].tolist() == [1, 1]
    assert provider.calls[-1][1].tolist() == [0, 1]


def test_train_grounding_selection_is_balanced_and_reproducible(tmp_path):
    from grounding_evaluation import ValGroundingProvider

    for class_name in ('n00000001', 'n00000002'):
        class_dir = tmp_path / 'train' / class_name
        class_dir.mkdir(parents=True)
        for index in range(8):
            (class_dir / f'{index}.JPEG').touch()

    provider = ValGroundingProvider(tmp_path, 32, 2, 'train', per_class=3, selection_seed=11)
    repeated = ValGroundingProvider(tmp_path, 32, 2, 'train', per_class=3, selection_seed=11)
    different = ValGroundingProvider(tmp_path, 32, 2, 'train', per_class=3, selection_seed=12)
    assert provider.files == repeated.files
    assert provider.files != different.files
    assert all(len(files) == 3 for files in provider.files)
    assert all(files == sorted(files) for files in provider.files)
    assert provider.selection_record()['per_class'] == 3
    assert provider.selection_record()['selection_seed'] == 11


def test_noised_real_grounding_matches_training(args):
    from noised_grounding_evaluation import NoisedRealGroundingSampler, generate_noised_grounding

    args.condition_mode_pixel = 'noised'

    class Provider:
        num_classes = args.class_num

        def load_batch(self, labels, intra_indices):
            return torch.ones(len(labels), 3, 32, 32)

    model = denoiser.Denoiser(args).eval()
    model.steps_pixel = 1
    sampler = NoisedRealGroundingSampler(model, Provider(), num_images=8, batch_size=3)
    labels = torch.tensor([0, 0, 1])
    torch.manual_seed(11)
    z_cond = sampler(labels)
    torch.manual_seed(11)
    expected_t = .75 + .25 * torch.rand(len(labels))
    clean = model.encode_grounding(torch.ones(len(labels), 3, 32, 32))
    expected_z = expected_t.view(-1, 1, 1, 1) * clean
    expected_z += (1 - expected_t.view(-1, 1, 1, 1)) * torch.randn_like(clean)
    torch.testing.assert_close(z_cond, expected_z)

    torch.manual_seed(23)
    expected_z = sampler(labels)
    model.net.calls.clear()
    torch.manual_seed(23)
    images = generate_noised_grounding(model, sampler, model.generate, labels)
    assert images.shape == (3, 3, 32, 32)
    assert len(model.net.calls) == 2
    for tokens, _ in model.net.calls:
        torch.testing.assert_close(tokens, model._as_tokens(expected_z))
    assert not model.dino_net.calls


def test_fixed_noised_real_grounding(args):
    from noised_grounding_evaluation import NoisedRealGroundingSampler

    args.condition_mode_pixel = 'noised'

    class Provider:
        num_classes = args.class_num

        def load_batch(self, labels, intra_indices):
            return torch.ones(len(labels), 3, 32, 32)

    model = denoiser.Denoiser(args).eval()
    sampler = NoisedRealGroundingSampler(
        model, Provider(), num_images=8, batch_size=3, fixed_t_dino_pixel=.75)
    labels = torch.tensor([0, 0, 1])
    torch.manual_seed(5)
    z_cond = sampler(labels)
    torch.manual_seed(5)
    clean = model.encode_grounding(torch.ones(len(labels), 3, 32, 32))
    expected_z = .75 * clean + .25 * torch.randn_like(clean)
    torch.testing.assert_close(z_cond, expected_z)


def test_pred_real_grounding_matches_training(args):
    from noised_grounding_evaluation import NoisedRealGroundingSampler, generate_noised_grounding

    args.condition_mode_pixel = 'pred'

    class Provider:
        num_classes = args.class_num

        def load_batch(self, labels, intra_indices):
            return torch.ones(len(labels), 3, 32, 32)

    model = denoiser.Denoiser(args).eval()
    model.steps_pixel = 1
    sampler = NoisedRealGroundingSampler(
        model, Provider(), num_images=8, batch_size=3, fixed_t_dino_pixel=.75)
    labels = torch.tensor([0, 0, 1])
    torch.manual_seed(9)
    z_cond = sampler(labels)
    torch.manual_seed(9)
    clean = model.encode_grounding(torch.ones(len(labels), 3, 32, 32))
    noised = .75 * clean + .25 * torch.randn_like(clean)
    expected_z = model.dino_net(noised, torch.full((len(labels),), .75), labels)
    torch.testing.assert_close(z_cond, expected_z)

    torch.manual_seed(10)
    expected_z = sampler(labels)
    model.net.calls.clear()
    torch.manual_seed(10)
    generate_noised_grounding(model, sampler, model.generate, labels)
    assert len(model.net.calls) == 2
    for tokens, _ in model.net.calls:
        torch.testing.assert_close(tokens, model._as_tokens(expected_z))


def test_label_dropout_masks_pixel_not_teacher(args):
    args.label_drop_prob = 1.0
    model = denoiser.Denoiser(args)
    x, labels = batch()
    model(x, labels)
    assert not model.net.calls[0][1].any()
    torch.testing.assert_close(model.dino_net.calls[0][2], labels)


def test_logit_normal_condition_schedule(args, monkeypatch):
    args.t_schedule_dino_pixel = 'logit_normal'
    model = denoiser.Denoiser(args)
    def sample(n, device, mean, std):
        assert (mean, std) == (args.t_mean_dino, args.t_std_dino)
        return torch.full((n,), .2, device=device)
    monkeypatch.setattr(denoiser, 'sample_logit_normal', sample)
    _, s = model.prepare_grounding_pixel(*batch())
    torch.testing.assert_close(s, torch.full_like(s, .2))


def test_dino_training_has_no_pixel_and_only_dino_gradients(args):
    args.loss_scale_dino = 1.0
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
    saved = vars(copy.copy(args))
    for key, legacy in LEGACY_KEYS.items():
        if key in saved:
            saved[legacy] = saved.pop(key)
    # Old joint EMA only includes trainable weights, not fixed positions/encoder.
    return {
        'args': type(args)(**saved), 'model': state,
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
    args.loss_scale_dino = 1.0
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
    args.sampling_method_pixel = args.sampling_method_dino = method
    model = denoiser.Denoiser(args).eval()
    def unexpected(*args):
        raise AssertionError('generation cannot encode a real image')
    monkeypatch.setattr(model.dino_encoder, 'encode', unexpected)
    images = model.generate(torch.tensor([0, 1]))
    assert images.shape == (2, 3, 32, 32)
    assert torch.isfinite(images).all()
    assert len(model.dino_net.calls) == (3 if method == 'heun' else 2)
    assert all(tokens is model.net.calls[0][0] for tokens, _ in model.net.calls)


def test_fail_fast_without_frozen_pretraining_or_missing_resume(tmp_path):
    args = get_args_parser().parse_args([])
    with pytest.raises(ValueError, match='requires --dino_checkpoint'):
        validate_args(args)
    args.resume = str(tmp_path / 'missing')
    with pytest.raises(FileNotFoundError, match='Resume'):
        validate_args(args)
    dino_args = main_dino.get_args_parser().parse_args([])
    assert main_dino.validate_args(dino_args) is None
    assert not hasattr(dino_args, 'online_eval')


@torch.compiler.disable
def test_actual_transformer_gradient_and_generation(args, monkeypatch):
    from model_jit import JiT
    from stage1.dino_dit import DinoDiT
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


def test_evaluate_keep_images(args, monkeypatch):
    import engine_jit

    model = denoiser.Denoiser(args)
    initialize_ema(model)
    args.num_images = 4
    args.keep_images = True

    class LogWriter:
        def add_scalar(self, *args):
            pass

    monkeypatch.setattr(engine_jit.torch_fidelity, 'calculate_metrics', lambda **kwargs: {
        'frechet_inception_distance': 1.0,
        'inception_score_mean': 2.0,
    })
    monkeypatch.setattr(engine_jit, '_cache_local_inception_weights', lambda path: None)
    monkeypatch.setattr(engine_jit, 'fid_stats_path', lambda size: 'unused')
    monkeypatch.setattr(engine_jit.shutil, 'rmtree', lambda path: (_ for _ in ()).throw(
        AssertionError('keep_images must not delete generated images')))
    engine_jit.evaluate(model, args, 1, batch_size=2, log_writer=LogWriter())
    assert len(list(Path(args.output_dir).rglob('*.png'))) == 4


@pytest.mark.parametrize('condition_mode', ['clean', 'noised', 'pred'])
def test_end_to_end_dino_then_pixel_and_resume_cpu(args, monkeypatch, tmp_path, condition_mode):
    import main_jit
    from util import training
    # Tiny models use 32px; keep the real validation except the production input size check.
    monkeypatch.setattr(main_jit, 'validate_dino_args', lambda args: None)
    monkeypatch.setattr(main_dino, 'validate_dino_args', lambda args: None)
    monkeypatch.chdir(tmp_path)
    images = torch.randint(0, 256, (4, 3, 32, 32), dtype=torch.uint8)
    dataset = torch.utils.data.TensorDataset(images, torch.tensor([0, 1, 2, 3]))
    monkeypatch.setattr(training.datasets, 'ImageFolder', lambda *a, **kw: dataset)
    args.num_workers = 0
    args.pin_mem = False
    args.batch_size = 2
    args.epochs = args.save_last_freq = args.save_interval = 1
    args.warmup_epochs = 0
    args.output_dir = ''
    args.condition_mode_pixel = condition_mode
    dino_args = main_dino.get_args_parser().parse_args([])
    for key in vars(dino_args).keys() & vars(args).keys() - {'training_stage'}:
        setattr(dino_args, key, getattr(args, key))
    main_dino.main(dino_args)
    teacher_path = Path('checkpoints/stage1-dinov3b-dinodit-b-e1/checkpoint-1.pth')
    dino_checkpoint = read_checkpoint(teacher_path)
    assert dino_checkpoint['training_stage'] == 'dino'
    assert not any(k.startswith('net.') for k in dino_checkpoint['model'])
    args.dino_checkpoint = str(teacher_path)
    main_jit.main(args)
    pixel_path = Path(f'checkpoints/stage1-dinov3b-dinodit-b-e1-stage2-pixeljit-b-dinodit-{condition_mode}-e1/checkpoint-last.pth')
    first = read_checkpoint(pixel_path)
    assert first['training_stage'] == 'pixel'
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
    args.t_schedule_dino_pixel = 'logit_normal'
    with pytest.raises(ValueError, match='t_schedule_dino_pixel'):
        resume_model(model, optimizer, checkpoint, args)
