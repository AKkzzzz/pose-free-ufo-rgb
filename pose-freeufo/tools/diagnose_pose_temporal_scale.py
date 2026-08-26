#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--data-root", required=True)
p.add_argument("--annotation-list", required=True)
p.add_argument("--scene-index", type=int, required=True)
p.add_argument("--omega-npz", required=True)
args = p.parse_args()

# load UFO scene annotation
lines = [
    x.strip() for x in Path(args.annotation_list).read_text().splitlines()
    if x.strip()
]
annotation_path = Path(args.data_root) / lines[args.scene_index]
scene = json.loads(annotation_path.read_text())

# load raw Omega poses BEFORE rig metric scaling
with np.load(args.omega_npz, allow_pickle=False) as x:
    frame_ids = x["frame_ids"].astype(int)
    camera_ids = x["camera_ids"].astype(str)
    omega_c2w = x["omega_c2w_raw"].astype(np.float64)

all_ratios = []

for cam in sorted(set(camera_ids)):
    idx = np.where(camera_ids == cam)[0]
    idx = idx[np.argsort(frame_ids[idx])]

    ratios = []

    for a, b in zip(idx[:-1], idx[1:]):
        f0 = int(frame_ids[a])
        f1 = int(frame_ids[b])

        # only consecutive timestamps
        if f1 != f0 + 1:
            continue

        C_omega0 = omega_c2w[a, :3, 3]
        C_omega1 = omega_c2w[b, :3, 3]

        gt0 = np.asarray(
            scene["camera_to_world"][cam][f0],
            dtype=np.float64
        )
        gt1 = np.asarray(
            scene["camera_to_world"][cam][f1],
            dtype=np.float64
        )

        C_gt0 = gt0[:3, 3]
        C_gt1 = gt1[:3, 3]

        d_omega = np.linalg.norm(C_omega1 - C_omega0)
        d_gt = np.linalg.norm(C_gt1 - C_gt0)

        if d_omega < 1e-6 or d_gt < 1e-3:
            continue

        ratios.append(d_gt / d_omega)

    ratios = np.asarray(ratios)

    print(
        f"camera {cam}: "
        f"median={np.median(ratios):.6f}, "
        f"mean={np.mean(ratios):.6f}, "
        f"std={np.std(ratios):.6f}, "
        f"N={len(ratios)}"
    )

    all_ratios.extend(ratios.tolist())

all_ratios = np.asarray(all_ratios)

print("\n==============================")
print(f"Omega temporal pose scale : {np.median(all_ratios):.6f}")
print(f"mean                      : {np.mean(all_ratios):.6f}")
print(f"std                       : {np.std(all_ratios):.6f}")
print("Omega depth metric scale  : 16.460003")
print("MoGe/Omega depth scale    : 17.628928")
print("Rig-baseline pose scale   : 12.079337")
print("==============================")
