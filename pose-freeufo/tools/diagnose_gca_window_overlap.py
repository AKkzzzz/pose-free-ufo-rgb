#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import numpy as np


def load_window(path):
    with np.load(path, allow_pickle=False) as x:
        return {
            "frame_ids": x["frame_ids"].astype(int),
            "camera_ids": x["camera_ids"].astype(str),
            "c2w": x["omega_c2w_rig_local"].astype(np.float64),
        }


def mapping(x):
    return {
        (int(f), str(c)): i
        for i, (f, c) in enumerate(zip(x["frame_ids"], x["camera_ids"]))
    }


def fit_rotation(src_R, dst_R):
    # dst_R ~= Rg @ src_R
    M = np.zeros((3,3))
    for a, b in zip(src_R, dst_R):
        M += b @ a.T

    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt

    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    return R


def apply_se3(poses, R, t):
    out = poses.copy()
    out[:, :3, :3] = np.einsum("ij,njk->nik", R, poses[:, :3, :3])
    out[:, :3, 3] = np.einsum("ij,nj->ni", R, poses[:, :3, 3]) + t
    return out


def rot_error_deg(A, B):
    rel = np.einsum("nij,nkj->nik", A, B)
    c = np.clip(
        (np.trace(rel, axis1=1, axis2=2) - 1) / 2,
        -1, 1
    )
    return np.degrees(np.arccos(c))


def sim3_scale(src_xyz, dst_xyz, R):
    xs = src_xyz - src_xyz.mean(0)
    ys = dst_xyz - dst_xyz.mean(0)

    xr = (R @ xs.T).T
    den = np.sum(xr * xr)

    if den < 1e-12:
        return np.nan

    return np.sum(xr * ys) / den


p = argparse.ArgumentParser()
p.add_argument("--left", type=Path, required=True)
p.add_argument("--right", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()

left = load_window(args.left)
right = load_window(args.right)

lm = mapping(left)
rm = mapping(right)

keys = sorted(set(lm) & set(rm))

if len(keys) < 3:
    raise RuntimeError(f"not enough overlap poses: {len(keys)}")

L = np.stack([left["c2w"][lm[k]] for k in keys])
R0 = np.stack([right["c2w"][rm[k]] for k in keys])

# Align RIGHT -> LEFT
Rg = fit_rotation(
    R0[:, :3, :3],
    L[:, :3, :3],
)

# Preserve GCA metric scale: SE3 only
offsets = (
    L[:, :3, 3]
    - np.einsum("ij,nj->ni", Rg, R0[:, :3, 3])
)

tg = np.median(offsets, axis=0)

aligned = apply_se3(R0, Rg, tg)

terr = np.linalg.norm(
    aligned[:, :3, 3] - L[:, :3, 3],
    axis=1,
)

rerr = rot_error_deg(
    aligned[:, :3, :3],
    L[:, :3, :3],
)

# Only diagnostic:
# if GCA metric scales agree, best Sim3 scale should be near 1.
s = sim3_scale(
    R0[:, :3, 3],
    L[:, :3, 3],
    Rg,
)

report = {
    "common_poses": len(keys),
    "common_frames": sorted(set(k[0] for k in keys)),
    "common_keys": [[int(f), str(c)] for f,c in keys],

    "se3_translation_median_m": float(np.median(terr)),
    "se3_translation_mean_m": float(np.mean(terr)),
    "se3_translation_max_m": float(np.max(terr)),

    "se3_rotation_median_deg": float(np.median(rerr)),
    "se3_rotation_mean_deg": float(np.mean(rerr)),
    "se3_rotation_max_deg": float(np.max(rerr)),

    "diagnostic_sim3_scale": float(s),
    "diagnostic_scale_error_percent": float(abs(s - 1) * 100),

    "alignment_rotation": Rg.tolist(),
    "alignment_translation": tg.tolist(),

    "scale_applied": False,
    "uses_gt_geometry": False,
}

args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2) + "\n")

print(json.dumps(report, indent=2))
