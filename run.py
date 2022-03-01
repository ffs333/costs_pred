import argparse

from core.pipeline import pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description='Script to run training process')

    parser.add_argument('-c', '--config', type=str,
                        required=True,
                        help='path to configuration file')

    parser.add_argument('-v', '--verbose', type=int,
                        required=False, const=1, nargs='?', default=1,
                        help='1 for show inputs, 0 for hide')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    path_to_config, verbose = args.config, args.verbose
    pipeline(path_to_config, verbose)