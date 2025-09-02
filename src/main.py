#!/usr/bin/env python3
"""
Tiny pipeline launcher - like a shortcut to run the three main stages in sequence or individually.
Each stage is implemented in its own script, this just imports and calls them.

Usage:
  python main.py                # default: all (prepare -> train -> webcam) with each script's defaults
  python main.py prepare        # only data preparation stage with defaults
  python main.py train          # only training stage with defaults
  python main.py webcam         # only inference stage with defaults
"""

import os, sys, argparse
HERE = os.path.abspath(os.path.dirname(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)  # ensure project root is importable (so `src` is a package)

def _run(stage: str, argv_tail):
    """Import the stage's script and run its main(), forwarding argv_tail."""
    saved = list(sys.argv)
    try:
        if stage == "prepare":
            from src.data.prepare_fer2013 import main as entry
        elif stage == "train":
            from src.train.train import main as entry
        elif stage == "webcam":
            from src.realtime.run_webcam import main as entry
        else:
            raise ValueError(f"Unknown stage: {stage}")

        # Rebuild argv for the child script: scriptname + forwarded args
        sys.argv = [f"{stage}.py"] + list(argv_tail)
        entry()
    finally:
        sys.argv = saved

def main():
    ap = argparse.ArgumentParser(prog="main", description="Tiny pipeline orchestrator.")
    ap.add_argument("stage", nargs="?", default="all", choices=["all", "prepare", "train", "webcam"],
                    help="Which stage to run (default: all).")
    ap.add_argument("stage_args", nargs=argparse.REMAINDER,
                    help="Arguments forwarded to the chosen stage. Put them after `--`.")
    args = ap.parse_args()

    # Strip leading `--` if present (argparse.REMAINDER keeps it)
    tail = args.stage_args[1:] if args.stage_args[:1] == ["--"] else args.stage_args

    if args.stage == "all":
        _run("prepare", [])
        _run("train", [])
        _run("webcam", [])
    else:
        _run(args.stage, tail)

if __name__ == "__main__":
    main()