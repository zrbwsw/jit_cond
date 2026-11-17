"""Exercise training/backprop and noise-only sampling with a real frozen DINOv3.

No ImageNet data is read. Training targets are synthetic images in [-1,1];
the DINO branch must receive their noised states. Sampling starts from noise.

Example (use local DINO paths/environment overrides):
  TORCHDYNAMO_DISABLE=1 python tools/smoke_train.py --tiny --device cuda \
      --grounding_repo /path/to/dinov3 --grounding_weights /path/to/dinov3_vitb16.pth

For weight-free CPU regressions instead:
  TORCHDYNAMO_DISABLE=1 OMP_NUM_THREADS=2 python -m unittest discover -s tests -v
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import os
from pathlib import Path
import sys

os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from denoiser import Denoiser
from main_jit import get_args_parser
from model_jit import JiT, JiT_models
from util.misc import add_weight_decay


def main():
    parser = argparse.ArgumentParser(description=__doc__, parents=[get_args_parser()])
    parser.add_argument('--tiny', action='store_true', help='Use a small JiT (32-pixel inputs) while keeping the real DINOv3-B encoder')
    parser.add_argument('--smoke_steps', type=int, default=3, help='At least 3 updates are needed to open JiT zero-initialized output/AdaLN paths')
    parser.add_argument('--save_samples', default='', help='Optional .pt destination for generated samples')
    parser.set_defaults(batch_size=2, num_sampling_steps=2, cfg=2.0, num_workers=0,
                        online_eval=False, label_drop_prob=0.0, ground_drop_prob=0.0)
    args = parser.parse_args()
    if args.smoke_steps < 3:
        parser.error('--smoke_steps must be at least 3 to verify projector updates')
    if not args.use_grounding:
        parser.error('this smoke test requires grounding; omit --no-use_grounding')
    if args.tiny:
        def tiny(**kwargs):
            return JiT(depth=2, hidden_size=64, num_heads=4, bottleneck_dim=16,
                       in_context_len=2, in_context_start=1, patch_size=16, **kwargs)
        JiT_models['smoke-tiny'] = tiny
        args.model, args.img_size = 'smoke-tiny', 32
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; pass --tiny --device cpu for a CPU smoke run')
    torch.manual_seed(args.seed)
    model = Denoiser(args).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    model.ema_params1 = [p.detach().clone() for p in trainable]
    model.ema_params2 = [p.detach().clone() for p in trainable]
    optimizer = torch.optim.AdamW(add_weight_decay(model, args.weight_decay), lr=args.lr or 1e-3)
    before = {n:p.detach().clone() for n,p in model.net.grounding_projector.named_parameters()}
    def amp():
        return torch.amp.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
    encoder_inputs = []
    capture = model.grounding_encoder.register_forward_pre_hook(lambda _m, a: encoder_inputs.append(a[0].detach().clone()))
    try:
        model.train()
        for step in range(args.smoke_steps):
            clean = torch.rand(args.batch_size, 3, args.img_size, args.img_size, device=device) * 2 - 1
            labels = torch.randint(args.class_num, (args.batch_size,), device=device)
            optimizer.zero_grad(set_to_none=True)
            with amp():
                loss = model(clean, labels)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            loss.backward()
            if any(p.grad is not None for p in model.grounding_encoder.parameters()):
                raise RuntimeError('Frozen DINO received gradients')
            if model.grounding_encoder.model.training:
                raise RuntimeError('DINO must stay in eval mode during JiT training')
            if torch.equal(encoder_inputs[-1], clean):
                raise RuntimeError('DINO received the clean target instead of the noisy state')
            projector_grad = sum(p.grad.detach().float().norm().item() for p in model.net.grounding_projector.parameters() if p.grad is not None)
            optimizer.step()
            model.update_ema()
            print(f'step={step+1} loss={loss.item():.6f} projector_grad_norm_sum={projector_grad:.6g}')
            encoder_inputs.clear()
        changed = [n for n,p in model.net.grounding_projector.named_parameters() if not torch.equal(before[n],p.detach())]
        if not changed:
            raise RuntimeError('No projector parameter updated after training')
        print(f'Projector parameters updated: {len(changed)}; frozen DINO has no gradients.')
        model.eval()
        with torch.no_grad(), amp():
            samples = model.generate(labels)
        if not torch.isfinite(samples).all() or samples.shape != clean.shape:
            raise RuntimeError('Invalid generated sample tensor')
        expected = args.num_sampling_steps if args.sampling_method == 'euler' else 2*args.num_sampling_steps-1
        if len(encoder_inputs) != expected:
            raise RuntimeError(f'Expected {expected} current-state DINO evaluations, got {len(encoder_inputs)}')
        print(f'Noise-only {args.sampling_method} sampling passed: {tuple(samples.shape)}, DINO evaluations={len(encoder_inputs)}')
        if args.save_samples:
            destination = Path(args.save_samples).expanduser()
            destination.parent.mkdir(parents=True, exist_ok=True)
            torch.save(samples.detach().cpu(), destination)
            print(f'Saved samples to {destination}')
    finally:
        capture.remove()


if __name__ == '__main__':
    main()
