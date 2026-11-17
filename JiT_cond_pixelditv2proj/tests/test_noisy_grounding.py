"""CPU regressions for noisy-state DINO conditioning and time-conditioned projection.

Run from the project directory:
  TORCHDYNAMO_DISABLE=1 OMP_NUM_THREADS=2 python -m unittest discover -s tests -v
No pretrained weights, image dataset, GPU or network access is needed.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn
import torch.nn.functional as F

import denoiser
import grounding
import model_jit
from pixeldit2_projector import TimestepProjection
import util.misc as misc


class TinyDINO(nn.Module):
    """Deterministic patch encoder with real parameters to check freezing."""
    embed_dim = 8
    patch_size = 16

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, self.embed_dim)
        self.inputs = []

    def forward_features(self, images):
        self.inputs.append(images.detach().clone())
        patches = F.avg_pool2d(images, self.patch_size).flatten(2).transpose(1, 2)
        return {"x_norm_patchtokens": self.proj(patches)}


def tiny_factory(patch_size=16, **kwargs):
    return model_jit.JiT(
        depth=2, hidden_size=32, num_heads=4, bottleneck_dim=8,
        in_context_len=2, in_context_start=1, patch_size=patch_size, **kwargs,
    )


def make_args(**updates):
    args = dict(
        model="test-tiny", img_size=32, class_num=3, attn_dropout=0.0,
        proj_dropout=0.0, use_grounding=True, grounding_dim=8,
        grounding_model="dinov3_vitb16", grounding_weights="", grounding_repo="",
        grounding_proj_blocks=1, grounding_proj_heads=None,
        label_drop_prob=0.0, P_mean=-0.8, P_std=0.8, t_eps=0.05,
        noise_scale=1.0, ground_drop_prob=0.0, ema_decay1=0.9, ema_decay2=0.8,
        sampling_method="heun", num_sampling_steps=3, cfg=2.0,
        interval_min=0.0, interval_max=1.0,
    )
    args.update(updates)
    return SimpleNamespace(**args)


def make_model(patch_size=16, **updates):
    with mock.patch.dict(denoiser.JiT_models, {
        "test-tiny": lambda **kw: tiny_factory(patch_size, **kw),
    }), mock.patch.object(grounding, "load_dino_v3", side_effect=lambda *a, **kw: TinyDINO()):
        return denoiser.Denoiser(make_args(**updates))


def activate_paths(net):
    """Release JiT's deliberate zero output/gates so all gradient paths are visible."""
    with torch.no_grad():
        nn.init.normal_(net.final_layer.linear.weight, std=0.03)
        for block in net.blocks:
            nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.03)
        nn.init.normal_(net.final_layer.adaLN_modulation[-1].weight, std=0.03)
        for block in net.grounding_projector.blocks:
            nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.1)


class NoisyGroundingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        torch.set_num_threads(2)

    def test_train_encodes_noisy_state_and_passes_identical_time(self):
        model = make_model().train()
        clean = torch.rand(2, 3, 32, 32) * 2 - 1
        times = torch.tensor([0.2, 0.7])
        noise = torch.full_like(clean, -0.75)
        expected = times[:, None, None, None] * clean + (1-times[:, None, None, None]) * noise
        encoded, net_calls, proj_times = [], [], []
        handles = [
            model.grounding_encoder.register_forward_pre_hook(lambda _m, a: encoded.append(a[0].detach().clone())),
            model.net.register_forward_pre_hook(lambda _m, a, k: net_calls.append((a, k)), with_kwargs=True),
            model.net.grounding_projector.register_forward_pre_hook(lambda _m, a: proj_times.append(a[1].detach().clone())),
        ]
        try:
            with mock.patch.object(model, "sample_t", return_value=times), mock.patch.object(torch, "randn_like", return_value=noise):
                loss = model(clean, torch.tensor([0, 1]))
        finally:
            for h in handles: h.remove()
        self.assertEqual(len(encoded), 1)
        torch.testing.assert_close(encoded[0], expected)
        self.assertFalse(torch.equal(encoded[0], clean))
        torch.testing.assert_close(net_calls[0][0][0], expected)
        torch.testing.assert_close(net_calls[0][0][1], times)
        torch.testing.assert_close(proj_times[0], times)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))

    def test_training_label_dropout_disables_grounding_on_same_samples(self):
        model = make_model().train()
        calls = []
        hook = model.net.register_forward_pre_hook(lambda _m, a, k: calls.append((a, k)), with_kwargs=True)
        try:
            with mock.patch.object(model, "drop_labels_with_mask", return_value=(torch.tensor([3, 1]), torch.tensor([False, True]))):
                model(torch.randn(2, 3, 32, 32), torch.tensor([0, 1]))
        finally:
            hook.remove()
        torch.testing.assert_close(calls[0][0][2], torch.tensor([3, 1]))
        torch.testing.assert_close(calls[0][1]["grounding_mask"], torch.tensor([False, True]))

    def test_cfg_null_branch_has_no_grounding_and_interval_is_respected(self):
        model = make_model(cfg=3.0, interval_min=0.1, interval_max=0.9).eval()
        z = torch.randn(2, 3, 32, 32)
        t = torch.tensor([0.5, 0.05]).view(2, 1, 1, 1)
        labels = torch.tensor([0, 1])
        calls = []
        def predict(state, times, class_labels, **kwargs):
            calls.append((state.clone(), times.clone(), class_labels.clone(), kwargs))
            velocity = 0.5 if bool((class_labels == model.num_classes).all()) else 2.0
            return state + (1-times[:, None, None, None]) * velocity
        with mock.patch.object(model.net, "forward", side_effect=predict), mock.patch.object(model, "encode_grounding", wraps=model.encode_grounding) as encoder:
            velocity = model._forward_sample(z, t, labels)
        self.assertEqual(encoder.call_count, 1)
        self.assertEqual(len(calls), 2)
        self.assertIsNotNone(calls[0][3]["grounding_tokens"])
        self.assertIs(calls[1][3]["grounding_mask"], False)
        torch.testing.assert_close(calls[1][2], torch.full_like(labels, 3))
        torch.testing.assert_close(velocity[0], torch.full_like(z[0], 5.0))
        torch.testing.assert_close(velocity[1], torch.full_like(z[1], 2.0))

    def test_heun_reencodes_predictor_state_at_next_time(self):
        model = make_model().eval()
        z = torch.randn(1, 3, 32, 32)
        t, t_next = torch.tensor([0.1]).view(1, 1, 1, 1), torch.tensor([0.4]).view(1, 1, 1, 1)
        encoded, proj_times = [], []
        h1 = model.grounding_encoder.register_forward_pre_hook(lambda _m, a: encoded.append(a[0].clone()))
        h2 = model.net.grounding_projector.register_forward_pre_hook(lambda _m, a: proj_times.append(a[1].clone()))
        try:
            result = model._heun_step(z, t, t_next, torch.tensor([1]))
        finally:
            h1.remove(); h2.remove()
        # The untouched JiT output layer predicts zero, so v(z,t) = -z / (1-t).
        predictor = z + (t_next-t) * (-z / (1-t))
        self.assertEqual(len(encoded), 2)
        torch.testing.assert_close(encoded[0], z)
        torch.testing.assert_close(encoded[1], predictor)
        self.assertFalse(torch.equal(encoded[0], encoded[1]))
        torch.testing.assert_close(proj_times[0], t.flatten())
        torch.testing.assert_close(proj_times[1], t_next.flatten())
        self.assertEqual(result.shape, z.shape)

    def test_generation_uses_noise_only_euler_and_heun(self):
        labels = torch.tensor([0])
        for method, evaluations in [("euler", 3), ("heun", 5)]:
            with self.subTest(method=method):
                model = make_model(sampling_method=method).eval()
                with mock.patch.object(model, "encode_grounding", wraps=model.encode_grounding) as encoder:
                    sample = model.generate(labels)
                self.assertEqual(encoder.call_count, evaluations)
                self.assertEqual(sample.shape, (1, 3, 32, 32))
                self.assertTrue(torch.isfinite(sample).all())
                with self.assertRaises(TypeError):
                    model.generate(labels, clean_images=torch.zeros_like(sample))
                with self.assertRaises(TypeError):
                    model.generate(labels, grounding_tokens=torch.zeros(1, 4, 8))

    def test_dino_preprocess_does_not_clip_noisy_values_and_matches_patch32_grid(self):
        model = make_model(patch_size=32, img_size=64)
        model.train()
        self.assertFalse(model.grounding_encoder.model.training)
        self.assertTrue(all(not p.requires_grad for p in model.grounding_encoder.parameters()))
        z = torch.full((2, 3, 64, 64), 3.0)
        tokens = model.encode_grounding(z)
        self.assertEqual(tokens.shape, (2, 4, 8))
        # Target JiT grid=2x2, so DINO/16 sees 32x32, not the raw 64x64.
        captured = model.grounding_encoder.model.inputs[-1]
        self.assertEqual(captured.shape, (2, 3, 32, 32))
        expected = (torch.tensor(2.0) - model.grounding_encoder.image_mean) / model.grounding_encoder.image_std
        torch.testing.assert_close(captured, expected.expand_as(captured))
        self.assertFalse(tokens.requires_grad)
        output = model.net(z, torch.tensor([0.3, 0.6]), torch.tensor([0, 1]), grounding_tokens=tokens)
        self.assertEqual(output.shape, z.shape)
        self.assertEqual(model.net.grounding_projector.input_proj.in_features, 8)
        self.assertEqual(model.net.grounding_projector.output_proj.out_features, 32)
        self.assertFalse(hasattr(model.net, "grounding_proj"))

    def test_projector_is_trainable_dino_frozen_optimizer_and_ema_cover_it(self):
        model = make_model().train()
        activate_paths(model.net)
        trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        model.ema_params1 = [p.detach().clone() for _, p in trainable]
        model.ema_params2 = [p.detach().clone() for _, p in trainable]
        optimizer = torch.optim.AdamW(misc.add_weight_decay(model, 0.01), lr=1e-3)
        optimizer_ids = {id(p) for group in optimizer.param_groups for p in group['params']}
        projector = model.net.grounding_projector
        self.assertTrue(all(id(p) in optimizer_ids for p in projector.parameters()))
        self.assertTrue(all(id(p) not in optimizer_ids for p in model.grounding_encoder.parameters()))
        initial = {name: p.detach().clone() for name, p in projector.named_parameters()}
        loss = model(torch.randn(2, 3, 32, 32), torch.tensor([0, 1]))
        loss.backward()
        self.assertTrue(all(p.grad is None for p in model.grounding_encoder.parameters()))
        for name, p in projector.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
        self.assertGreater(projector.input_proj.weight.grad.abs().sum().item(), 0)
        self.assertGreater(projector.output_proj.weight.grad.abs().sum().item(), 0)
        self.assertGreater(projector.t_embedder.mlp[0].weight.grad.abs().sum().item(), 0)
        optimizer.step()
        model.update_ema()
        self.assertFalse(torch.equal(initial['input_proj.weight'], projector.input_proj.weight))
        for i, (name, p) in enumerate(trainable):
            if name.startswith('net.grounding_projector.'):
                self.assertEqual(model.ema_params1[i].shape, p.shape)

    def test_patch_adaln_receives_time_plus_label_plus_projector_and_sample_mask(self):
        model = make_model().eval()
        net = model.net
        images = torch.randn(2, 3, 32, 32)
        tokens = model.encode_grounding(images)
        times, labels = torch.tensor([0.25, 0.65]), torch.tensor([0, 1])
        keep = torch.tensor([False, True])
        global_condition = net.t_embedder(times) + net.y_embedder(labels)
        projected = net.grounding_projector(tokens, times, (2, 2))
        expected = global_condition[:, None] + projected * keep[:, None, None]
        seen = {"block0": [], "block1": [], "final": []}
        handles = []
        for key, module in [("block0", net.blocks[0].adaLN_modulation),
                            ("block1", net.blocks[1].adaLN_modulation),
                            ("final", net.final_layer.adaLN_modulation)]:
            handles.append(module.register_forward_pre_hook(
                lambda _m, a, key=key: seen[key].append(a[0].detach().clone())))
        try:
            net(images, times, labels, grounding_tokens=tokens, grounding_mask=keep)
        finally:
            for handle in handles: handle.remove()
        torch.testing.assert_close(seen['block0'][0], expected)
        torch.testing.assert_close(seen['final'][0], expected)
        torch.testing.assert_close(seen['block1'][0][:, 2:], expected)
        torch.testing.assert_close(seen['block1'][0][:, :2], global_condition[:, None].expand(-1, 2, -1))
        self.assertGreater((expected[1, 0]-expected[1, 1]).abs().max().item(), 0)
        torch.testing.assert_close(expected[0], global_condition[0].expand(4, -1))
        # A whole-batch unconditional request bypasses all projector work.
        with mock.patch.object(net.grounding_projector, 'forward', wraps=net.grounding_projector.forward) as project:
            net(images, times, labels, grounding_tokens=tokens, grounding_mask=False)
        project.assert_not_called()

    def test_checkpoint_roundtrip_contains_projector_and_ema(self):
        model = make_model()
        trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        model.ema_params1 = [p.detach().clone().add_(0.01) for _, p in trainable]
        model.ema_params2 = [p.detach().clone().add_(0.02) for _, p in trainable]
        optimizer = torch.optim.AdamW(misc.add_weight_decay(model))
        with tempfile.TemporaryDirectory() as temp:
            misc.save_model(SimpleNamespace(output_dir=temp), model, optimizer, epoch=4, epoch_name='test')
            checkpoint = torch.load(Path(temp)/'checkpoint-test.pth', weights_only=False, map_location='cpu')
        expected = {n for n, _ in trainable}
        self.assertEqual(set(checkpoint['model_ema1']), expected)
        self.assertEqual(set(checkpoint['model_ema2']), expected)
        self.assertTrue(any('grounding_projector' in k for k in expected))
        self.assertFalse(any(k.startswith('grounding_encoder.') for k in expected))
        restored = make_model()
        restored.load_state_dict(checkpoint['model'])
        for name, value in model.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], value)
        for i, (name, _) in enumerate(trainable):
            torch.testing.assert_close(checkpoint['model_ema1'][name], model.ema_params1[i])


class ProjectorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)
        torch.set_num_threads(2)

    def test_pixeldit_initialization_dimension_mapping_and_learned_time_dependence(self):
        projector = TimestepProjection(in_dim=8, out_dim=32, hidden_size=32, num_blocks=1, num_heads=4)
        tokens = torch.randn(2, 4, 8)
        t0, t1 = torch.zeros(2), torch.ones(2)
        block = projector.blocks[0]
        self.assertEqual(torch.count_nonzero(block.adaLN_modulation[-1].weight), 0)
        self.assertEqual(torch.count_nonzero(block.adaLN_modulation[-1].bias), 0)
        self.assertGreater(torch.count_nonzero(projector.output_proj.weight), 0)
        initial = projector(tokens, t0, (2, 2))
        torch.testing.assert_close(initial, projector.output_proj(projector.input_proj(tokens)))
        torch.testing.assert_close(initial, projector(tokens, t1, (2, 2)))
        self.assertEqual(initial.shape, (2, 4, 32))
        # PixelDiT zero-initializes AdaLN, so time dependence becomes visible once learned gates open.
        with torch.no_grad():
            nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.15)
        early = projector(tokens, t0, (2, 2))
        late = projector(tokens, t1, (2, 2))
        self.assertGreater((early-late).abs().max().item(), 1e-5)


if __name__ == '__main__':
    unittest.main()
