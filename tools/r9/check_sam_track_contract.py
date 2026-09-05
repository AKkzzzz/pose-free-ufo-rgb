#!/usr/bin/env python3
"""Exercise the frozen clean-R9 SAM loader with real scene names."""

import argparse
import json
import re
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ufo.dataset.sam_tracks import load_sam_track_mask


CAMERAS = (1, 0, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/UFO_paper")
    parser.add_argument("--sam-root", type=Path, default=ROOT / "data/r9_sam_tracks")
    parser.add_argument("--scene-index", action="append", type=int)
    parser.add_argument("--all-scenes", action="store_true")
    args = parser.parse_args()
    lines = (args.data_root / "scene_list/waymo_train.txt").read_text().splitlines()
    scene_indices = range(798) if args.all_scenes else (args.scene_index or [621])
    checked = 0
    for scene_index in scene_indices:
        annotation = Path(lines[scene_index])
        if not annotation.is_absolute():
            annotation = args.data_root / annotation
        scene_name = json.loads(annotation.read_text())["scene_name"]
        namespaces = []
        image_dir = args.data_root / "datasets/waymo/training" / f"{scene_index:03d}" / "images"
        for slot, camera in enumerate(CAMERAS):
            pattern = re.compile(rf"^(\d+)_{camera}\.jpg$")
            frames = sorted(
                int(match.group(1)) for path in image_dir.iterdir()
                if (match := pattern.match(path.name))
            )
            if not frames:
                raise RuntimeError(f"No RGB frames: scene={scene_name} camera={camera}")
            foreground = set()
            for frame in sorted({frames[0], frames[len(frames) // 2], frames[-1]}):
                mask = load_sam_track_mask(
                    args.sam_root, scene_name, camera, frame, (160, 240), camera_slot=slot
                )
                if mask.dtype != torch.int64 or tuple(mask.shape) != (160, 240):
                    raise RuntimeError(f"Loader contract mismatch: {mask.dtype} {tuple(mask.shape)}")
                unique = torch.unique(mask)
                if int(unique[0]) != 0:
                    raise RuntimeError("Background ID 0 is missing")
                foreground.update(int(value) for value in unique[unique > 0].tolist())
                checked += 1
            lower, upper = (slot + 1) * 10000, (slot + 2) * 10000
            if foreground and not all(lower < value < upper for value in foreground):
                raise RuntimeError(f"Camera namespace violation: slot={slot}")
            namespaces.append(foreground)
        for left in range(3):
            for right in range(left + 1, 3):
                if not namespaces[left].isdisjoint(namespaces[right]):
                    raise RuntimeError("Camera namespaces overlap")
        print(f"PASS index={scene_index} scene_name={scene_name} cameras=1,0,2")
    print(f"R9_SAM_TRACK_CONTRACT=PASS masks_checked={checked}")


if __name__ == "__main__":
    main()
