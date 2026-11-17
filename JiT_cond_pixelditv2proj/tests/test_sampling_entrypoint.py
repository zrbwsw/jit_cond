"""Verify the real sampling entrypoint never requires a training/validation image tree."""
from contextlib import nullcontext, redirect_stdout
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from test_noisy_grounding import make_model, TinyDINO, tiny_factory
import grounding
import denoiser
import main_jit
import util.misc as misc


class SamplingEntrypointTests(unittest.TestCase):
    @staticmethod
    def init_cpu(args):
        args.distributed = False
        args.rank = 0
        args.world_size = 1
        args.gpu = 0

    def test_evaluate_gen_loads_projector_ema_and_never_constructs_dataset(self):
        torch.manual_seed(41)
        model = make_model(num_sampling_steps=2)
        params = [p for p in model.parameters() if p.requires_grad]
        model.ema_params1 = [p.detach().clone() for p in params]
        model.ema_params2 = [p.detach().clone() for p in params]
        optimizer = torch.optim.AdamW(misc.add_weight_decay(model))
        with tempfile.TemporaryDirectory() as temp:
            misc.save_model(SimpleNamespace(output_dir=temp), model, optimizer, epoch=0, epoch_name='test')
            checkpoint = Path(temp)/'checkpoint-test.pth'
            missing_data = Path(temp)/'definitely-no-image-directory'
            args = main_jit.get_args_parser().parse_args([
                '--evaluate_gen', '--checkpoint', str(checkpoint), '--data_path', str(missing_data),
                '--device', 'cpu', '--model', 'test-tiny', '--img_size', '32', '--class_num', '3',
                '--grounding_dim', '8', '--gen_bsz', '1', '--num_images', '3',
                '--num_sampling_steps', '2', '--output_dir', str(Path(temp)/'out'),
            ])
            generated = []
            def evaluate(loaded_model, _args, _epoch, **kwargs):
                loaded_model.eval()
                names = [n for n,p in loaded_model.named_parameters() if p.requires_grad]
                self.assertTrue(any('grounding_projector' in n for n in names))
                self.assertEqual(len(loaded_model.ema_params1), len(names))
                generated.append(loaded_model.generate(torch.tensor([0])))
            with mock.patch.object(main_jit.misc, 'init_distributed_mode', side_effect=self.init_cpu), \
                 mock.patch.object(main_jit.datasets, 'ImageFolder', side_effect=AssertionError('Sampling attempted to read images')) as dataset, \
                 mock.patch.object(main_jit, 'SummaryWriter'), \
                 mock.patch.object(main_jit, 'evaluate', side_effect=evaluate), \
                 mock.patch.object(main_jit.torch.random, 'fork_rng', side_effect=lambda: nullcontext()), \
                 mock.patch.dict(denoiser.JiT_models, {'test-tiny': tiny_factory}), \
                 mock.patch.object(grounding, 'load_dino_v3', side_effect=lambda *a, **k: TinyDINO()), \
                 redirect_stdout(io.StringIO()):
                main_jit.main(args)
            dataset.assert_not_called()
            self.assertFalse(missing_data.exists())
            self.assertEqual(len(generated), 1)
            self.assertEqual(generated[0].shape, (1,3,32,32))
            self.assertTrue(torch.isfinite(generated[0]).all())

    def test_sampling_missing_checkpoint_fails_before_dataset_or_model_creation(self):
        args = main_jit.get_args_parser().parse_args(['--evaluate_gen', '--device', 'cpu'])
        with mock.patch.object(main_jit.misc, 'init_distributed_mode', side_effect=self.init_cpu), \
             mock.patch.object(main_jit.datasets, 'ImageFolder') as dataset, \
             mock.patch.object(main_jit, 'Denoiser') as factory, \
             redirect_stdout(io.StringIO()):
            with self.assertRaises(FileNotFoundError):
                main_jit.main(args)
        dataset.assert_not_called()
        factory.assert_not_called()


if __name__ == '__main__':
    unittest.main()
