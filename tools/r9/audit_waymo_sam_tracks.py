#!/usr/bin/env python3
"""Audit all requested SAM pairs directly against annotation-derived RGB frames."""

import argparse
import json
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CAMERAS = ("1", "0", "2")


def load_records(data_root):
    lines = (data_root / "scene_list/waymo_train.txt").read_text().splitlines()
    if len(lines) != 798:
        raise RuntimeError(f"Expected 798 annotations, found {len(lines)}")
    records = []
    for index, item in enumerate(lines):
        path = Path(item)
        if not path.is_absolute():
            path = data_root / path
        payload = json.loads(path.read_text())
        records.append((index, payload["scene_name"]))
    return records


def indices_for(data_root, index, camera):
    image_dir = data_root / "datasets/waymo/training" / f"{index:03d}" / "images"
    pattern = re.compile(rf"^(\d+)_{camera}\.jpg$")
    return sorted(
        int(match.group(1))
        for path in image_dir.iterdir()
        if (match := pattern.match(path.name))
    )


def audit_pair(data_root, sam_root, index, scene_name, camera):
    frames = indices_for(data_root, index, camera)
    pair = sam_root / scene_name / camera
    marker = pair / ".r9_sam2_done.json"
    mask_dir = pair / "mask_data"
    errors = []
    expected = {f"mask_{frame:06d}.npy" for frame in frames}
    actual = {path.name for path in mask_dir.glob("mask_*.npy")} if mask_dir.is_dir() else set()
    if not frames:
        errors.append("no RGB frames")
    if not marker.is_file():
        errors.append("missing done marker")
    else:
        try:
            record = json.loads(marker.read_text())
            if record.get("num_frames") != len(frames):
                errors.append("marker num_frames mismatch")
            if record.get("mask_count", len(frames)) != len(frames):
                errors.append("marker mask_count mismatch")
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"invalid marker: {error}")
    if actual != expected:
        errors.append(f"mask set mismatch missing={len(expected-actual)} extra={len(actual-expected)}")
    for name in sorted(expected & actual):
        try:
            mask = np.load(mask_dir / name, mmap_mode="r", allow_pickle=False)
            if mask.ndim != 2 or mask.size == 0:
                errors.append(f"{name}: expected nonempty 2D mask")
                continue
            converted = mask.astype(np.int64)
            if np.any(mask != converted):
                errors.append(f"{name}: unsafe int64 conversion")
            elif int(converted.min()) < 0:
                errors.append(f"{name}: negative ID")
            elif int(converted.max()) >= 10000:
                errors.append(f"{name}: max local ID >= 10000")
            elif not np.any(converted == 0):
                errors.append(f"{name}: background 0 missing")
        except Exception as error:
            errors.append(f"{name}: {error!r}")
        if len(errors) >= 20:
            break
    return {"scene": scene_name, "numeric_scene": f"{index:03d}", "camera": camera,
            "expected_masks": len(frames), "actual_masks": len(actual),
            "complete": not errors, "errors": errors}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/UFO_paper")
    parser.add_argument("--sam-root", type=Path, default=ROOT / "data/r9_sam_tracks")
    parser.add_argument("--scene-index", action="append", type=int)
    parser.add_argument("--require-full", action="store_true")
    args = parser.parse_args()
    records = load_records(args.data_root)
    if args.scene_index:
        selected = set(args.scene_index)
        records = [record for record in records if record[0] in selected]
    results = [
        audit_pair(args.data_root, args.sam_root, index, scene, camera)
        for index, scene in records for camera in CAMERAS
    ]
    failures = [result for result in results if not result["complete"]]
    summary = {
        "expected_pairs": len(results),
        "complete_pairs": len(results) - len(failures),
        "incomplete_pairs": len(failures),
        "failed_pairs": len(failures),
        "total_expected_masks": sum(result["expected_masks"] for result in results),
        "total_actual_masks": sum(result["actual_masks"] for result in results),
        "failures": failures,
    }
    manifest = args.sam_root / "full_audit.json"
    temporary = manifest.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(manifest)
    for key in ("expected_pairs", "complete_pairs", "incomplete_pairs", "failed_pairs"):
        print(f"{key}={summary[key]}")
    for failure in failures[:10]:
        print(f"FAIL {failure['scene']} camera={failure['camera']} {failure['errors'][:3]}")
    expected = 2394 if args.require_full else len(results)
    if len(results) != expected or failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
