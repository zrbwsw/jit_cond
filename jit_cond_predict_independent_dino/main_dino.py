"""Train only DinoDiT; Pixel JiT is never instantiated."""

from main_jit import get_args_parser, main


if __name__ == '__main__':
    parser = get_args_parser()
    parser.set_defaults(training_stage='dino', output_dir='./checkpoints/dino')
    main(parser.parse_args())
