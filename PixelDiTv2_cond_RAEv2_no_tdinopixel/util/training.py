"""JiT input pipeline, distributed setup, AdamW and epoch checkpoints."""
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from util import misc


def resolve_checkpoint(value):
    if not value:
        return None
    path = Path(value).expanduser()
    if path.is_dir():
        path /= 'checkpoint-last.pth'
    if not path.is_file():
        raise FileNotFoundError(f'Pixel checkpoint missing: {path}')
    return path


def initialize_run(args):
    misc.init_distributed_mode(args)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    return torch.device(args.device)


def make_train_loader(args):
    from torchvision import datasets, transforms
    from util.crop import center_crop_arr
    transform = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.PILToTensor(),
    ])
    dataset = datasets.ImageFolder(Path(args.data_path) / 'train', transform=transform)
    sampler = torch.utils.data.DistributedSampler(
        dataset, num_replicas=misc.get_world_size(), rank=misc.get_rank(), shuffle=True, seed=args.seed)
    return torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=args.batch_size,
                                       num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True)


def make_log_writer(output_dir):
    if misc.is_main_process():
        from torch.utils.tensorboard import SummaryWriter
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return SummaryWriter(log_dir=output_dir)
    return None


def wrap_distributed(model, args):
    if not args.distributed:
        return model
    ids = [args.gpu] if torch.device(args.device).type == 'cuda' else None
    # Frozen encoders are identical across processes. No running batch statistics.
    return torch.nn.parallel.DistributedDataParallel(model, device_ids=ids, broadcast_buffers=False,
                                                       gradient_as_bucket_view=True)


def make_optimizer(model, args):
    params = [p for p in model.parameters() if p.requires_grad]
    global_batch = args.batch_size * args.accum_steps * misc.get_world_size()
    print(f'Trainable parameters: {sum(p.numel() for p in params):,}; effective global batch: {global_batch}')
    return torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)


def save_epoch(args, model, optimizer, completed_epoch):
    from checkpoint_io import build_checkpoint
    if not misc.is_main_process():
        return
    names = []
    if completed_epoch % args.save_last_freq == 0 or completed_epoch == args.epochs:
        names.append('last')
    if completed_epoch % args.save_interval == 0:
        names.append(str(completed_epoch))
    if names:
        checkpoint = build_checkpoint(args, model, optimizer, completed_epoch)
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        for name in names:
            with tempfile.NamedTemporaryFile(dir=output, prefix='.checkpoint-', suffix='.tmp', delete=False) as handle:
                tmp = Path(handle.name)
            try:
                torch.save(checkpoint, tmp)
                os.replace(tmp, output / f'checkpoint-{name}.pth')
            finally:
                tmp.unlink(missing_ok=True)
