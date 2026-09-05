#!/usr/bin/env python3
"""Alias fully valid legacy numeric SAM pairs under their real scene names."""

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CAMERAS = ("1", "0", "2")


def annotations(data_root):
    lines = (data_root / "scene_list/waymo_train.txt").read_text().splitlines()
    if len(lines) != 798:
        raise RuntimeError(f"Expected 798 annotations, found {len(lines)}")
    for index, item in enumerate(lines):
        path = Path(item)
        if not path.is_absolute():
            path = data_root / path
        payload = json.loads(path.read_text())
        yield index, payload["scene_name"]


def frame_indices(data_root, numeric_scene, camera):
    image_dir = data_root / "datasets/waymo/training" / numeric_scene / "images"
    pattern = re.compile(rf"^(\d+)_{camera}\.jpg$")
    return sorted(
        int(match.group(1))
        for path in image_dir.iterdir()
        if (match := pattern.match(path.name))
    )


def valid_pair(pair, indices):
    marker = pair / ".r9_sam2_done.json"
    mask_dir = pair / "mask_data"
    if not marker.is_file() or not mask_dir.is_dir() or not indices:
        return False
    try:
        record = json.loads(marker.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if record.get("num_frames") != len(indices):
        return False
    actual = list(mask_dir.glob("mask_*.npy"))
    if len(actual) != len(indices):
        return False
    for frame in indices:
        path = mask_dir / f"mask_{frame:06d}.npy"
        if not path.is_file():
            return False
        try:
            mask = np.load(path, mmap_mode="r", allow_pickle=False)
            if mask.ndim != 2 or mask.size == 0:
                return False
            if not np.issubdtype(mask.dtype, np.number):
                return False
            if np.any(mask != mask.astype(np.int64)):
                return False
            if int(mask.min()) < 0 or int(mask.max()) >= 10000 or not np.any(mask == 0):
                return False
        except (OSError, ValueError, TypeError, OverflowError):
            return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/UFO_paper")
    parser.add_argument("--sam-root", type=Path, default=ROOT / "data/r9_sam_tracks")
    parser.add_argument("--scene-index", type=int)
    args = parser.parse_args()
    imported = skipped = kept = invalid_existing = 0
    for index, scene_name in annotations(args.data_root):
        if args.scene_index is not None and index != args.scene_index:
            continue
        numeric = f"{index:03d}"
        for camera in CAMERAS:
            source = args.sam_root / numeric / camera
            indices = frame_indices(args.data_root, numeric, camera)
            scene_dir = args.sam_root / scene_name
            alias = scene_dir / camera
            if alias.exists() or alias.is_symlink():
                if alias.is_symlink() and alias.resolve() == source.resolve():
                    if valid_pair(source, indices):
                        imported += 1
                    else:
                        invalid_existing += 1
                        print(
                            f"INVALID_EXISTING scene={scene_name} camera={camera} "
                            f"path={alias} action=leave_for_preprocess"
                        )
                    continue
                if alias.is_dir() and not alias.is_symlink():
                    if valid_pair(alias, indices):
                        kept += 1
                        print(
                            f"KEEP_EXISTING scene={scene_name} camera={camera} "
                            f"path={alias}"
                        )
                    else:
                        invalid_existing += 1
                        print(
                            f"INVALID_EXISTING scene={scene_name} camera={camera} "
                            f"path={alias} action=leave_for_preprocess"
                        )
                    continue
                invalid_existing += 1
                print(
                    f"INVALID_EXISTING scene={scene_name} camera={camera} "
                    f"path={alias} action=leave_for_preprocess"
                )
                continue
            if not valid_pair(source, indices):
                skipped += 1
                continue
            scene_dir.mkdir(exist_ok=True)
            relative = os.path.relpath(source, scene_dir)
            alias.symlink_to(relative, target_is_directory=True)
            imported += 1
    print(
        f"SAM_NUMERIC_IMPORT imported_pairs={imported} skipped_pairs={skipped} "
        f"kept_existing_pairs={kept} invalid_existing_pairs={invalid_existing}"
    )


if __name__ == "__main__":
    main()
