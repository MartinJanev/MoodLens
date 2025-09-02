# src/scripts/train.py
import os, sys
HERE = os.path.dirname(__file__)
SRC_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from src.train.train_config import CONFIG
from src.train.train_loop import run_training

def main():
    run_training(CONFIG)

if __name__ == '__main__':
    main()
