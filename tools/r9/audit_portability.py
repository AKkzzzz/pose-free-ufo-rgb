#!/usr/bin/env python3
"""Reject runtime paths that escape the only H200-visible yx-ufo tree."""

import os
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VISIBLE_ROOT = ROOT.parent.resolve()
FORBIDDEN = ("/inspire/hdd3", "/inspire/hdd2", "/inspire/hdd/project")


def inside_visible(path):
    try:
        path.resolve(strict=True).relative_to(VISIBLE_ROOT)
        return True
    except (FileNotFoundError, ValueError):
        return False


def main():
    failures = []
    required = (
        ROOT / "data/UFO_paper",
        ROOT / "data/r9_sam_tracks",
        ROOT / "third_party",
        ROOT / "outputs",
    )
    for path in required:
        if not path.is_symlink() or not inside_visible(path):
            failures.append(f"invalid required symlink: {path} -> {path.resolve(strict=False)}")

    scan_roots = (ROOT / "scripts/r9", ROOT / "tools/r9", ROOT / "configs/h200")
    for scan_root in scan_roots:
        for path in scan_root.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".sh", ".json", ".env"}:
                if path.resolve() == Path(__file__).resolve():
                    continue
                text = path.read_text(errors="ignore")
                for prefix in FORBIDDEN:
                    if prefix in text:
                        failures.append(f"forbidden text {prefix}: {path}")

    env_root = ROOT / "third_party/groundedsam2_env"
    metadata = [env_root / "pyvenv.cfg"]
    metadata += list(env_root.glob("lib/python*/site-packages/*.pth"))
    metadata += list(env_root.glob("lib/python*/site-packages/__editable__*finder.py"))
    for path in metadata:
        if path.is_file():
            text = path.read_text(errors="ignore")
            for prefix in FORBIDDEN:
                if prefix in text:
                    failures.append(f"forbidden runtime metadata {prefix}: {path}")

    cache = ROOT / "third_party/hf_cache"
    for directory, dirnames, filenames in os.walk(cache):
        for name in dirnames + filenames:
            path = Path(directory) / name
            if path.is_symlink() and not inside_visible(path):
                failures.append(f"HF cache symlink escapes visible root: {path}")
    train_list = ROOT / "data/UFO_paper/scene_list/waymo_train.txt"
    lines = train_list.read_text().splitlines()
    if len(lines) != 798:
        failures.append(f"Waymo train list has {len(lines)} lines")
    if any(Path(line).is_absolute() for line in lines):
        failures.append("Waymo train list contains absolute annotation paths")
    waymo = ROOT / "data/UFO_paper/datasets/waymo"
    if not inside_visible(waymo):
        failures.append(f"Waymo data escapes visible root: {waymo.resolve(strict=False)}")
    for index, item in enumerate(lines):
        annotation = ROOT / "data/UFO_paper" / item
        if not annotation.is_file():
            failures.append(f"missing local annotation: {annotation}")
            continue
        payload = json.loads(annotation.read_text())
        for camera in ("1", "0", "2"):
            for relative_image in payload["relative_image_path"][camera]:
                image = waymo / relative_image
                if not image.is_file():
                    failures.append(f"missing local RGB: {image}")
                    break
        instance_dir = waymo / "training" / f"{index:03d}" / "instances"
        for name in ("instances_info.json", "frame_instances.json"):
            if not (instance_dir / name).is_file():
                failures.append(f"missing local instance metadata: {instance_dir / name}")
    if failures:
        print("\n".join(failures))
        raise SystemExit(2)
    print(f"PORTABILITY_AUDIT=PASS visible_root={VISIBLE_ROOT}")


if __name__ == "__main__":
    main()
