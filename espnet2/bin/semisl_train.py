#!/usr/bin/env python3
from espnet2.tasks.semisl import SemiSLTask


def get_parser():
    parser = SemiSLTask.get_parser()
    return parser


def main(cmd=None):
    r"""ASR training.

    Example:

        % python asr_train.py asr --print_config --optim adadelta \
                > conf/train_asr.yaml
        % python asr_train.py --config conf/train_asr.yaml
    """
    SemiSLTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
