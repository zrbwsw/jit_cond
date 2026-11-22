"""两阶段通用的分布式训练设施。"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from resource_paths import imagenet_root
from util.crop import center_crop_arr
from util import misc


def add_training_args(parser, *, epochs):
    parser.add_argument('--img_size', default=256, type=int)
    parser.add_argument('--class_num', default=1000, type=int)
    parser.add_argument('--attn_dropout', default=0.0, type=float)
    parser.add_argument('--proj_dropout', default=0.0, type=float)
    parser.add_argument('--t_eps', default=5e-2, type=float)
    parser.add_argument('--label_drop_prob', default=0.1, type=float)
    parser.add_argument('--epochs', default=epochs, type=int)
    parser.add_argument('--warmup_epochs', default=5, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=None, type=float)
    parser.add_argument('--blr', default=5e-5, type=float)
    parser.add_argument('--min_lr', default=0., type=float)
    parser.add_argument('--lr_schedule', default='constant')
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--ema_decay1', default=0.9999, type=float)
    parser.add_argument('--ema_decay2', default=0.9996, type=float)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--pin_mem', action='store_true', dest='pin_mem')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--data_path', default=str(imagenet_root()))
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--resume', default='', help='Checkpoint from the same stage')
    parser.add_argument('--save_last_freq', default=5, type=int)
    parser.add_argument('--save_interval', default=20, type=int)
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', '--local-rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')


def validate_training_args(args):
    if min(args.epochs, args.batch_size, args.save_last_freq, args.save_interval, args.log_freq) < 1:
        raise ValueError('epochs, batch_size and save/log frequencies must be positive')


def resolve_checkpoint(path):
    if not path:
        return None
    checkpoint = Path(path).expanduser()
    if checkpoint.is_dir():
        checkpoint /= 'checkpoint-last.pth'
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Resume/evaluation checkpoint not found: {checkpoint}')
    return checkpoint


def initialize_run(args):
    misc.init_distributed_mode(args)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.optimize_ddp = False
    print('Arguments:\n{}'.format(args).replace(', ', ',\n'))
    return torch.device(args.device), seed


def make_train_loader(args, dataset=None):
    transform = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.PILToTensor(),
    ])
    if dataset is None:
        dataset = datasets.ImageFolder(str(Path(args.data_path) / 'train'), transform=transform)
    print(dataset)
    sampler = torch.utils.data.DistributedSampler(
        dataset, num_replicas=misc.get_world_size(), rank=misc.get_rank(), shuffle=True)
    return torch.utils.data.DataLoader(
        dataset, sampler=sampler, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=True)


def make_log_writer(output_dir):
    if misc.is_main_process():
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return SummaryWriter(log_dir=output_dir)
    return None


def wrap_distributed(model, args):
    if args.distributed:
        return torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    return model


def make_optimizer(model, args):
    global_batch = args.batch_size * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * global_batch / 256
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {num_params / 1e6:.6f}M, global batch: {global_batch}, LR: {args.lr:.2e}')
    return torch.optim.AdamW(misc.add_weight_decay(model, args.weight_decay), lr=args.lr, betas=(0.9, 0.95))


def save_epoch(args, model, optimizer, completed_epoch):
    if completed_epoch % args.save_last_freq == 0 or completed_epoch == args.epochs:
        misc.save_model(args, model, optimizer, completed_epoch, epoch_name='last')
    if completed_epoch % args.save_interval == 0:
        misc.save_model(args, model, optimizer, completed_epoch)
