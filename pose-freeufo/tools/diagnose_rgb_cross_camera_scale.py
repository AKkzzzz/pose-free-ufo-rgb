#!/usr/bin/env python3
"""
RGB-only same-timestamp cross-camera metric scale diagnostic.

Important:
  This estimates the metric scale of Omega cross-camera translation
  (rig baseline gauge). It MUST NOT be used directly as the temporal
  trajectory scale because Omega temporal and cross-camera translations
  have already been observed to use different effective gauges.

Inputs:
  - RGB images only
  - Omega raw camera poses / predicted intrinsics
  - MoGe-2 RGB metric depth

No GT pose / intrinsics / depth / camera_to_ego is used.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from diagnose_rgb_pose_scale import (
    epipolar_filter,
    load_gray,
    mutual_sift_matches,
    robust_log_scale,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--omega-npz", type=Path, required=True)
    p.add_argument("--moge-repo", type=Path, required=True)
    p.add_argument("--moge-model", type=Path, required=True)

    p.add_argument(
        "--camera-pairs",
        nargs="+",
        default=["0-1", "0-2", "1-2"],
        help="Unordered same-timestamp camera pairs.",
    )
    p.add_argument("--resolution-level", type=int, default=9)
    p.add_argument("--ratio", type=float, default=0.75)
    p.add_argument("--epi-threshold", type=float, default=2.0)

    p.add_argument("--min-depth", type=float, default=1.0)
    p.add_argument("--max-depth", type=float, default=80.0)
    p.add_argument(
        "--min-ray-baseline-sin2",
        type=float,
        default=1e-4,
        help="Reject rays almost parallel to relative translation.",
    )

    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def parse_camera_pairs(raw_pairs):
    pairs = []
    for raw in raw_pairs:
        fields = raw.split("-")
        if len(fields) != 2:
            raise ValueError(f"invalid camera pair {raw!r}; expected e.g. 0-1")
        a, b = fields
        if a == b:
            raise ValueError(f"camera pair must contain different cameras: {raw}")
        pairs.append((a, b))
    return pairs


def main():
    args = parse_args()

    manifest = json.loads(args.manifest.read_text())
    H, W = map(int, manifest["ufo_image_size"])

    entries = {
        (int(e["frame_id"]), str(e["camera_id"])): e
        for e in manifest["images"]
    }

    with np.load(args.omega_npz, allow_pickle=False) as x:
        frame_ids = x["frame_ids"].astype(int)
        camera_ids = x["camera_ids"].astype(str)
        c2w = x["omega_c2w_raw"].astype(np.float64)
        Ks = x["predicted_intrinsics_ufo"].astype(np.float64)

    lookup = {
        (int(frame), str(cam)): i
        for i, (frame, cam) in enumerate(zip(frame_ids, camera_ids))
    }

    camera_pairs = parse_camera_pairs(args.camera_pairs)

    # ---------- MoGe ----------
    sys.path.insert(0, str(args.moge_repo))
    from moge.model.v2 import MoGeModel

    device = torch.device("cuda")
    print("[MoGe] loading model", flush=True)
    moge = MoGeModel.from_pretrained(args.moge_model).eval().to(device)

    depth_cache = {}
    mask_cache = {}

    def get_metric_depth(frame, cam):
        key = (frame, cam)
        if key in depth_cache:
            return depth_cache[key], mask_cache[key]

        path = Path(entries[key]["path"])
        rgb_np = np.asarray(Image.open(path).convert("RGB")).copy()

        rgb = (
            torch.from_numpy(rgb_np)
            .float()
            .permute(2, 0, 1)
            .to(device)
            / 255.0
        )

        with torch.inference_mode():
            out = moge.infer(
                rgb,
                resolution_level=args.resolution_level,
                use_fp16=True,
                apply_mask=False,
            )

        depth = out["depth"].float().cpu().numpy()
        mask = out["mask"].cpu().numpy().astype(np.uint8)

        depth = cv2.resize(
            depth, (W, H), interpolation=cv2.INTER_LINEAR
        )
        mask = cv2.resize(
            mask, (W, H), interpolation=cv2.INTER_NEAREST
        ).astype(bool)

        depth_cache[key] = depth
        mask_cache[key] = mask
        return depth, mask

    # ---------- solve ----------
    reports = []
    all_scales = []

    def solve_direction(
        frame,
        src_cam,
        dst_cam,
        src_points,
        dst_points,
        sift_mutual,
    ):
        src_idx = lookup[(frame, src_cam)]
        dst_idx = lookup[(frame, dst_cam)]

        C0 = c2w[src_idx]
        C1 = c2w[dst_idx]
        K0 = Ks[src_idx]
        K1 = Ks[dst_idx]

        # X_dst = R X_src + s * t_raw
        T_1_0 = np.linalg.inv(C1) @ C0
        R = T_1_0[:3, :3]
        t = T_1_0[:3, 3]

        t2 = float(t @ t)
        if t2 < 1e-14:
            return None

        keep_epi = epipolar_filter(
            src_points,
            dst_points,
            K0,
            K1,
            R,
            t,
            args.epi_threshold,
        )

        p0 = src_points[keep_epi]
        p1 = dst_points[keep_epi]

        depth0, mask0 = get_metric_depth(frame, src_cam)

        invK0 = np.linalg.inv(K0)
        invK1 = np.linalg.inv(K1)

        scales = []
        samples = []

        for uv0, uv1 in zip(p0, p1):
            x = int(round(uv0[0]))
            y = int(round(uv0[1]))

            if x < 0 or x >= W or y < 0 or y >= H:
                continue
            if not mask0[y, x]:
                continue

            depth = float(depth0[y, x])
            if (
                not np.isfinite(depth)
                or depth < args.min_depth
                or depth > args.max_depth
            ):
                continue

            ray0 = invK0 @ np.array(
                [uv0[0], uv0[1], 1.0], dtype=np.float64
            )
            ray0 /= ray0[2]
            X0 = ray0 * depth

            b1 = invK1 @ np.array(
                [uv1[0], uv1[1], 1.0], dtype=np.float64
            )
            b1 /= np.linalg.norm(b1)

            a = np.cross(b1, t)
            c = np.cross(b1, R @ X0)

            denom = float(a @ a)

            # Equivalent to sin^2(angle(ray, baseline)).
            if denom / t2 < args.min_ray_baseline_sin2:
                continue

            scale = -float(a @ c) / denom

            if (
                not np.isfinite(scale)
                or scale <= 0.0
                or scale > 1e4
            ):
                continue

            scales.append(scale)
            samples.append((X0, uv1))

        pair_scale, log_mad, robust_keep = robust_log_scale(scales)
        if pair_scale is None:
            return None

        filtered_samples = [
            sample
            for sample, keep
            in zip(samples, robust_keep)
            if keep
        ]

        reproj = []
        for X0, uv1 in filtered_samples:
            X1 = R @ X0 + pair_scale * t
            if X1[2] <= 1e-6:
                continue

            q = K1 @ X1
            uv_pred = q[:2] / q[2]
            reproj.append(
                float(np.linalg.norm(uv_pred - uv1))
            )

        return {
            "frame": int(frame),
            "source_camera": src_cam,
            "target_camera": dst_cam,
            "direction": f"{src_cam}->{dst_cam}",
            "sift_mutual": int(sift_mutual),
            "epipolar_inliers": int(keep_epi.sum()),
            "scale_candidates": int(len(scales)),
            "scale_inliers": int(robust_keep.sum()),
            "raw_baseline_norm": float(np.sqrt(t2)),
            "scale": float(pair_scale),
            "log_mad": float(log_mad),
            "median_reprojection_px": (
                float(np.median(reproj)) if reproj else None
            ),
        }

    for cam_a, cam_b in camera_pairs:
        common_frames = sorted(
            frame
            for frame, cam in lookup
            if cam == cam_a
            and (frame, cam_b) in lookup
            and (frame, cam_a) in entries
            and (frame, cam_b) in entries
        )

        for frame in common_frames:
            path_a = Path(entries[(frame, cam_a)]["path"])
            path_b = Path(entries[(frame, cam_b)]["path"])

            # Cross-camera overlap is small, so match at native image
            # resolution instead of the 240x160 UFO resolution.
            img_a = cv2.imread(str(path_a), cv2.IMREAD_GRAYSCALE)
            img_b = cv2.imread(str(path_b), cv2.IMREAD_GRAYSCALE)

            if img_a is None:
                raise FileNotFoundError(path_a)
            if img_b is None:
                raise FileNotFoundError(path_b)

            ha, wa = img_a.shape
            hb, wb = img_b.shape

            pa_native, pb_native = mutual_sift_matches(
                img_a, img_b, args.ratio
            )

            if len(pa_native) < 8:
                continue

            # Omega intrinsics and MoGe depth below live in UFO resolution.
            # Current Waymo RGB manifests contain the full uncropped
            # 480x320 images, so native pixels map linearly to UFO pixels.
            pa = pa_native.copy()
            pb = pb_native.copy()

            pa[:, 0] *= W / float(wa)
            pa[:, 1] *= H / float(ha)
            pb[:, 0] *= W / float(wb)
            pb[:, 1] *= H / float(hb)

            # Solve both directions. This also checks MoGe consistency.
            for src, dst, psrc, pdst in (
                (cam_a, cam_b, pa, pb),
                (cam_b, cam_a, pb, pa),
            ):
                row = solve_direction(
                    frame,
                    src,
                    dst,
                    psrc,
                    pdst,
                    len(pa),
                )

                if row is None:
                    continue

                reports.append(row)
                all_scales.append(row["scale"])

                reproj = row["median_reprojection_px"]
                reproj_text = (
                    f"{reproj:.3f}px"
                    if reproj is not None else "nan"
                )

                print(
                    f"frame={frame:03d} "
                    f"cam={row['direction']:>4s} | "
                    f"matches={row['epipolar_inliers']:4d} | "
                    f"scale={row['scale']:8.3f} | "
                    f"logMAD={row['log_mad']:.3f} | "
                    f"reproj={reproj_text}",
                    flush=True,
                )

    global_scale, global_log_mad, global_keep = robust_log_scale(
        all_scales
    )

    per_direction = {}
    for direction in sorted({r["direction"] for r in reports}):
        vals = [
            r["scale"]
            for r in reports
            if r["direction"] == direction
        ]
        if vals:
            per_direction[direction] = {
                "median_scale": float(np.median(vals)),
                "num_estimates": len(vals),
            }

    report = {
        "method": (
            "moge_metric_depth_same_timestamp_"
            "cross_camera_omega_translation_scale"
        ),
        "scale_scope": "cross_camera_translation_only",
        "warning": (
            "Do not use this scale directly for Omega temporal "
            "trajectory translation."
        ),
        "camera_pairs": [
            f"{a}-{b}" for a, b in camera_pairs
        ],
        "num_estimates": len(reports),
        "global_scale": global_scale,
        "global_log_mad": global_log_mad,
        "global_inliers": (
            int(global_keep.sum()) if len(all_scales) else 0
        ),
        "per_direction": per_direction,
        "estimates": reports,
        "uses_gt_geometry": False,
        "uses_gt_intrinsics": False,
        "uses_gt_pose": False,
        "uses_camera_to_ego": False,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n"
    )

    print("\n================ RESULT ================")
    print("cross-camera scale :", global_scale)
    print("log MAD            :", global_log_mad)
    print("estimates          :", len(reports))
    print("per direction      :", per_direction)
    print("scope              : cross-camera only")
    print("GT geometry        : NOT USED")
    print("========================================")


if __name__ == "__main__":
    main()
