"""Launcher regressions; all SSH and training checks use --dry-run."""
from pathlib import Path
import os
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]


class LauncherTests(unittest.TestCase):
    def run_bash(self, *args, **env_updates):
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env.update(env_updates)
        return subprocess.run(
            ["bash", *args], cwd=ROOT, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True,
        ).stdout

    def test_train_and_sample_default_to_gpu1(self):
        for name in ("train_jit_b16_grounding_8gpu.sh", "sample_jit_b16_grounding_8gpu.sh"):
            source = (ROOT / "sh" / name).read_text()
            self.assertIn('GPU_HOST="${GPU_HOST:-agent-gpu-l2-1}"', source)

    def test_experiment_defaults_to_80_epochs(self):
        train = (ROOT / "sh/train_jit_b16_grounding_8gpu.sh").read_text()
        sample = (ROOT / "sh/sample_jit_b16_grounding_8gpu.sh").read_text()
        main = (ROOT / "main_jit.py").read_text()
        self.assertIn('EXPERIMENT="${EXPERIMENT:-JiT-B16-DINOv3B-Noisy-PixelDiT2Proj-B80}"', train)
        self.assertIn('EXPERIMENT="${EXPERIMENT:-JiT-B16-DINOv3B-Noisy-PixelDiT2Proj-B80}"', sample)
        self.assertIn('--epochs "${EPOCHS:-80}"', train)
        self.assertIn('DIST_PORT="${DIST_PORT:-29401}"', train)
        self.assertIn('--rdzv_endpoint "127.0.0.1:$DIST_PORT"', train)
        self.assertIn("JiT-B16-DINOv3B-Noisy-PixelDiT2Proj-B80", main)

    def test_run_on_gpu_dry_run_uses_current_cluster_mapping(self):
        output = self.run_bash("sh/run_on_gpu.sh", "--dry-run", "agent-gpu-l2-2", "nvidia-smi")
        self.assertIn("agent-gpu-l2-2-84043580-cfcd2", output)
        self.assertIn("h.pjlab.org.cn", output)
        self.assertIn("nvidia-smi", output)

    def test_local_single_gpu_train_dry_run(self):
        output = self.run_bash(
            "sh/train_jit_b16_grounding_8gpu.sh", "--dry-run",
            GPU_HOST="local", CUDA_VISIBLE_DEVICES="0", GPUS="1",
            BATCH_SIZE="2", ONLINE_EVAL="0",
        )
        self.assertIn("--nproc_per_node=1", output)
        self.assertIn("--batch_size 2", output)
        self.assertIn("--epochs 80", output)
        self.assertNotIn("--online_eval", output)

    def test_evaluation_uses_completed_epoch_count(self):
        source = (ROOT / "main_jit.py").read_text()
        self.assertIn(
            "if args.online_eval and ((epoch + 1) % args.eval_freq == 0 or epoch + 1 == args.epochs):",
            source,
        )
        selected = [epoch for epoch in range(80) if (epoch + 1) % 20 == 0 or epoch + 1 == 80]
        self.assertEqual(selected, [19, 39, 59, 79])

    def test_checkpoint_schedule_uses_completed_epoch_count(self):
        source = (ROOT / "main_jit.py").read_text()
        self.assertIn("if (epoch + 1) % args.save_last_freq == 0 or (epoch + 1) == args.epochs:", source)
        self.assertIn("if (epoch + 1) % args.save_interval == 0:", source)
        self.assertIn("epoch=(epoch + 1)", source)
        self.assertIn("--save_interval", source)
        last = [epoch for epoch in range(80) if (epoch + 1) % 5 == 0 or epoch + 1 == 80]
        persistent = [epoch for epoch in range(600) if (epoch + 1) % 20 == 0]
        self.assertEqual(last, list(range(4, 80, 5)))
        self.assertEqual(persistent, list(range(19, 600, 20)))
        self.assertIn("checkpoint['epoch'] if has_completed_epoch_count else checkpoint['epoch'] + 1", source)


if __name__ == "__main__":
    unittest.main()
