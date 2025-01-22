import argparse
import os

from eval import eval
from train import train
from utilities import getConfig

if __name__ == "__main__":
    os.environ["KMP_WARNINGS"] = "0"

    parser = argparse.ArgumentParser(description="Main Function UniversalEPI")
    parser.add_argument("--config_dir", type=str, help="Root directory for configs.")
    parser.add_argument("--mode", type=str, help="train or test", default="test")

    args = parser.parse_args()
    config = getConfig(args.config_dir)
    mode = args.mode

    assert mode in ["train", "test"], "Mode must be one of ['train', 'test']"

    if mode == "train":
        train(config)
    else:
        eval(config, config.cell_lines_test)
