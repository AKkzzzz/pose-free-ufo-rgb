#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def robust_ratio(target, source, valid=None):
    target = np.asarray(target, np.float64)
    source = np.asarray(source, np.float64)

    mask = (
        np.isfinite(target) &
        np.isfinite(source) &
        (target > 0.5) &
        (target < 120.0) &
        (source > 1e-5)
    )
    if valid is not None:
        mask &= valid

    h, w = mask.shape
    border = np.zeros_like(mask)
    border[int(.05*h):int(.95*h), int(.05*w):int(.95*w)] = True
    mask &= border

    log_ratio = np.log(target[mask]) - np.log(source[mask])
    med = np.median(log_ratio)

    return float(np.exp(med)), float(np.median(np.abs(log_ratio-med))), int(mask.sum())


p = argparse.ArgumentParser()
p.add_argument("--manifest", required=True)
p.add_argument("--omega-repo", required=True)
p.add_argument("--omega-checkpoint", required=True)
p.add_argument("--moge-repo", required=True)
p.add_argument("--moge-model", required=True)
p.add_argument("--max-images", type=int, default=6)
args = p.parse_args()

manifest = json.load(open(args.manifest))
entries = manifest["images"][:args.max_images]
paths = [e["path"] for e in entries]

device = "cuda"

# ---------- Omega ----------
sys.path.insert(0, args.omega_repo)
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images

imgs = load_and_preprocess_images(paths, image_resolution=512).to(device)

omega = VGGTOmega().eval().to(device)
state = torch.load(args.omega_checkpoint, map_location="cpu", weights_only=True)
omega.load_state_dict(state)
del state

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    pred = omega(imgs)

omega_depths = pred["depth"][0].float().cpu()
omega_confs = pred["depth_conf"][0].float().cpu()

# VGGT-Omega depth is [N,H,W,1]; remove the singleton channel.
if omega_depths.ndim == 4 and omega_depths.shape[-1] == 1:
    omega_depths = omega_depths[..., 0]

del omega, pred, imgs
torch.cuda.empty_cache()

# ---------- MoGe ----------
sys.path.insert(0, args.moge_repo)
from moge.model.v2 import MoGeModel

moge = MoGeModel.from_pretrained(args.moge_model).eval().to(device)

rows = []

for i, (e, path) in enumerate(zip(entries, paths)):
    rgb_np = np.asarray(Image.open(path).convert("RGB")).copy()
    h, w = rgb_np.shape[:2]

    rgb = torch.from_numpy(rgb_np).float().permute(2,0,1).to(device) / 255.

    with torch.inference_mode():
        out = moge.infer(rgb, resolution_level=9, use_fp16=True, apply_mask=True)

    moge_depth = out["depth"].float().cpu().numpy()
    moge_mask = out.get("mask")
    if moge_mask is not None:
        moge_mask = moge_mask.cpu().numpy().astype(bool)

    # Omega -> original image resolution
    od = F.interpolate(
        omega_depths[i][None,None],
        size=(h,w),
        mode="bilinear",
        align_corners=False,
    )[0,0].numpy()

    oc = F.interpolate(
        omega_confs[i][None,None],
        size=(h,w),
        mode="bilinear",
        align_corners=False,
    )[0,0].numpy()

    conf_mask = oc >= np.median(oc)

    # UFO / Waymo GT metric depth
    gt_path = Path(path.replace("images", "depth_flows")).with_suffix(".npy")
    if not gt_path.exists():
        raise FileNotFoundError(gt_path)

    arr = np.load(gt_path)
    gt_depth = arr[..., 0]

    if gt_depth.shape != (h,w):
        gt_depth = F.interpolate(
            torch.from_numpy(gt_depth).float()[None,None],
            size=(h,w),
            mode="nearest",
        )[0,0].numpy()

    valid = conf_mask
    if moge_mask is not None:
        valid &= moge_mask

    # A: Omega depth -> GT metric
    s_omega_depth, mad_a, n_a = robust_ratio(gt_depth, od, valid)

    # B: MoGe metric -> GT metric
    # ideal is 1.0
    s_moge_to_gt, mad_b, n_b = robust_ratio(gt_depth, moge_depth, valid)

    # Direct MoGe / Omega, should match previous ~17.6
    s_moge_omega, mad_c, n_c = robust_ratio(moge_depth, od, valid)

    row = {
        "frame": e["frame_id"],
        "camera": e["camera_id"],
        "omega_depth_to_gt": s_omega_depth,
        "moge_to_gt": s_moge_to_gt,
        "moge_over_omega": s_moge_omega,
    }
    rows.append(row)

    print(
        f"frame={int(e['frame_id']):03d} cam={e['camera_id']} | "
        f"GT/Omega={s_omega_depth:.3f} | "
        f"GT/MoGe={s_moge_to_gt:.3f} | "
        f"MoGe/Omega={s_moge_omega:.3f}"
    )

def gmed(key):
    return float(np.exp(np.median(np.log([r[key] for r in rows]))))

print("\n==============================")
print(f"Omega depth -> metric : {gmed('omega_depth_to_gt'):.6f}")
print(f"MoGe metric correction: {gmed('moge_to_gt'):.6f}  (ideal = 1.0)")
print(f"MoGe / Omega          : {gmed('moge_over_omega'):.6f}")
print(f"Omega pose rig oracle : 12.079337")
print("==============================")
