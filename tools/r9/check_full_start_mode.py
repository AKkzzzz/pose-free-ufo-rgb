#!/usr/bin/env python3
"""Report resume mode using only the current Full Waymo experiment directory."""

import argparse
from pathlib import Path


def current_checkpoints(run_dir):
    checkpoint_dir = run_dir / "checkpoints"
    return sorted(
        (
            path
            for path in checkpoint_dir.glob("*.pth")
            if "latest" not in path.name
        ),
        key=lambda path: path.stat().st_mtime,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    checkpoints = current_checkpoints(args.run_dir)
    if checkpoints:
        print("R9_FULL_START_MODE=AUTO_RESUME")
        print(f"R9_FULL_LATEST_CHECKPOINT={checkpoints[-1].resolve()}")
    else:
        print("R9_FULL_START_MODE=SCRATCH")


if __name__ == "__main__":
    main()
