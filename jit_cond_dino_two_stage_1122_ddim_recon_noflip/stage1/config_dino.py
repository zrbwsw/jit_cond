"""DinoDiT 网络与 latent 空间参数，Stage 2 加载 Stage 1 时沿用同一套定义。"""


def add_dino_args(parser):
    parser.add_argument('--grounding_dim', default=768, type=int)
    parser.add_argument('--dino_model', default='DinoDiT-B', choices=['DinoDiT-B'])
    parser.add_argument('--dino_encoder', default='dinov3b', choices=['dinov2b', 'dinov3b'])
    parser.add_argument('--dinov3_weights', default='')
    parser.add_argument('--dinov3_repo', default='')
    parser.add_argument('--dino_stat_path', default='')
    parser.add_argument('--t_mean_dino', '--D_mean', default=-1.2, type=float)
    parser.add_argument('--t_std_dino', '--D_std', default=1.0, type=float)
    parser.add_argument('--noise_scale_dino', default=1.0, type=float)
    parser.add_argument('--steps_dino', '--dino_steps', default=25, type=int)
    parser.add_argument('--sampling_method_dino', '--dino_sampling_method', default='heun', choices=['euler', 'heun'])
    parser.add_argument('--cfg_dino', '--dino_cfg', default=1.0, type=float)
    parser.add_argument('--interval_min_dino', '--dino_interval_min', default=0.0, type=float)
    parser.add_argument('--interval_max_dino', '--dino_interval_max', default=1.0, type=float)


def validate_dino_args(args):
    if (args.img_size, args.grounding_dim) != (256, 768):
        raise ValueError('The supplied DINO statistics require 256px images and 768x16x16 latents')
    if args.t_eps <= 0 or args.noise_scale_dino <= 0 or args.steps_dino < 1:
        raise ValueError('t_eps, noise_scale_dino and steps_dino must be positive')


def experiment_name_dino(encoder, model, completed_epochs):
    return f'stage1-{encoder}-{model.lower()}-e{completed_epochs}'
