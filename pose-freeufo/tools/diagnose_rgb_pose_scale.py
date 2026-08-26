#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--omega-npz", type=Path, required=True)
    p.add_argument("--moge-repo", type=Path, required=True)
    p.add_argument("--moge-model", type=Path, required=True)
    p.add_argument("--camera-id", default="0")
    p.add_argument("--gaps", type=int, nargs="+", default=[2, 4, 6])
    p.add_argument("--resolution-level", type=int, default=9)
    p.add_argument("--ratio", type=float, default=0.75)
    p.add_argument("--epi-threshold", type=float, default=2.0)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def skew(v):
    x, y, z = v
    return np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ], dtype=np.float64)


def mutual_sift_matches(img0, img1, ratio):
    sift = cv2.SIFT_create(nfeatures=5000)
    kp0, d0 = sift.detectAndCompute(img0, None)
    kp1, d1 = sift.detectAndCompute(img1, None)

    if d0 is None or d1 is None:
        return np.empty((0, 2)), np.empty((0, 2))

    bf = cv2.BFMatcher(cv2.NORM_L2)

    fwd = {}
    for pair in bf.knnMatch(d0, d1, k=2):
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            fwd[m.queryIdx] = m.trainIdx

    rev = {}
    for pair in bf.knnMatch(d1, d0, k=2):
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            rev[m.queryIdx] = m.trainIdx

    pairs = [(i, j) for i, j in fwd.items() if rev.get(j) == i]

    p0 = np.array([kp0[i].pt for i, _ in pairs], dtype=np.float64)
    p1 = np.array([kp1[j].pt for _, j in pairs], dtype=np.float64)
    return p0, p1


def epipolar_filter(p0, p1, K0, K1, R, t, threshold):
    if len(p0) == 0:
        return np.zeros(0, dtype=bool)

    E = skew(t) @ R
    F = np.linalg.inv(K1).T @ E @ np.linalg.inv(K0)

    x0 = np.concatenate([p0, np.ones((len(p0), 1))], axis=1)
    x1 = np.concatenate([p1, np.ones((len(p1), 1))], axis=1)

    Fx0 = (F @ x0.T).T
    Ftx1 = (F.T @ x1.T).T
    numer = np.abs(np.sum(x1 * Fx0, axis=1))

    d1 = numer / np.maximum(
        np.sqrt(Fx0[:, 0] ** 2 + Fx0[:, 1] ** 2), 1e-12
    )
    d0 = numer / np.maximum(
        np.sqrt(Ftx1[:, 0] ** 2 + Ftx1[:, 1] ** 2), 1e-12
    )

    return np.maximum(d0, d1) < threshold


def robust_log_scale(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0)]

    if len(values) < 5:
        return None, None, np.zeros(len(values), dtype=bool)

    logs = np.log(values)
    med = np.median(logs)
    mad = np.median(np.abs(logs - med))

    threshold = max(3.5 * 1.4826 * mad, 0.08)
    keep = np.abs(logs - med) <= threshold

    if keep.sum() < 5:
        return None, None, keep

    logs_in = logs[keep]
    scale = float(np.exp(np.median(logs_in)))
    log_mad = float(np.median(np.abs(logs_in - np.median(logs_in))))
    return scale, log_mad, keep


def load_gray(path, width, height):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def main():
    args = parse_args()

    manifest = json.loads(args.manifest.read_text())
    H, W = manifest["ufo_image_size"]

    entries = {
        (int(e["frame_id"]), str(e["camera_id"])): e
        for e in manifest["images"]
    }

    with np.load(args.omega_npz, allow_pickle=False) as x:
        frame_ids = x["frame_ids"].astype(int)
        camera_ids = x["camera_ids"].astype(str)
        c2w = x["omega_c2w_raw"].astype(np.float64)
        Ks = x["predicted_intrinsics_ufo"].astype(np.float64)

    lookup = {}
    for i, (frame, cam) in enumerate(zip(frame_ids, camera_ids)):
        lookup[(int(frame), str(cam))] = i

    frames = sorted(
        f for (f, c) in lookup
        if c == args.camera_id and (f, c) in entries
    )

    sys.path.insert(0, str(args.moge_repo))
    from moge.model.v2 import MoGeModel

    device = torch.device("cuda")
    print("[MoGe] loading model", flush=True)

    moge = MoGeModel.from_pretrained(args.moge_model).eval().to(device)

    depth_cache = {}
    mask_cache = {}

    def get_metric_depth(frame):
        if frame in depth_cache:
            return depth_cache[frame], mask_cache[frame]

        path = Path(entries[(frame, args.camera_id)]["path"])
        rgb_np = np.asarray(Image.open(path).convert("RGB")).copy()

        rgb = (
            torch.from_numpy(rgb_np)
            .float()
            .permute(2, 0, 1)
            .to(device) / 255.0
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

        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

        depth_cache[frame] = depth
        mask_cache[frame] = mask
        return depth, mask

    pair_reports = []
    all_pair_scales = []

    for gap in args.gaps:
        for f0 in frames:
            f1 = f0 + gap
            if f1 not in frames:
                continue

            i0 = lookup[(f0, args.camera_id)]
            i1 = lookup[(f1, args.camera_id)]

            C0 = c2w[i0]
            C1 = c2w[i1]
            K0 = Ks[i0]
            K1 = Ks[i1]

            T_1_0 = np.linalg.inv(C1) @ C0
            R = T_1_0[:3, :3]
            t = T_1_0[:3, 3]

            if np.linalg.norm(t) < 1e-8:
                continue

            path0 = Path(entries[(f0, args.camera_id)]["path"])
            path1 = Path(entries[(f1, args.camera_id)]["path"])

            img0 = load_gray(path0, W, H)
            img1 = load_gray(path1, W, H)

            p0, p1 = mutual_sift_matches(img0, img1, args.ratio)

            epi_keep = epipolar_filter(
                p0, p1, K0, K1, R, t, args.epi_threshold
            )
            p0 = p0[epi_keep]
            p1 = p1[epi_keep]

            depth0, mask0 = get_metric_depth(f0)

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

                d = float(depth0[y, x])
                if not np.isfinite(d) or d < 1.0 or d > 120.0:
                    continue

                ray0 = invK0 @ np.array([uv0[0], uv0[1], 1.0])
                ray0 /= ray0[2]
                X0 = ray0 * d

                b1 = invK1 @ np.array([uv1[0], uv1[1], 1.0])
                b1 /= np.linalg.norm(b1)

                a = np.cross(b1, t)
                c = np.cross(b1, R @ X0)

                denom = float(a @ a)
                if denom < 1e-12:
                    continue

                s = -float(a @ c) / denom

                if not np.isfinite(s) or s <= 0 or s > 1e4:
                    continue

                scales.append(s)
                samples.append((X0, uv1))

            pair_scale, log_mad, robust_keep = robust_log_scale(scales)

            if pair_scale is None:
                continue

            filtered_samples = [
                sample for sample, keep
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
                reproj.append(float(np.linalg.norm(uv_pred - uv1)))

            row = {
                "frame0": f0,
                "frame1": f1,
                "gap": gap,
                "sift_mutual": int(len(epi_keep)),
                "epipolar_inliers": int(epi_keep.sum()),
                "scale_candidates": int(len(scales)),
                "scale_inliers": int(robust_keep.sum()),
                "scale": pair_scale,
                "log_mad": log_mad,
                "median_reprojection_px": (
                    float(np.median(reproj)) if reproj else None
                ),
            }

            pair_reports.append(row)
            all_pair_scales.append(pair_scale)

            print(
                f"{f0:03d}->{f1:03d} gap={gap} | "
                f"matches={row['epipolar_inliers']:4d} | "
                f"scale={pair_scale:8.3f} | "
                f"reproj={row['median_reprojection_px']:.3f}px",
                flush=True,
            )

    global_scale, global_log_mad, global_keep = robust_log_scale(all_pair_scales)

    per_gap = {}
    for gap in args.gaps:
        vals = [r["scale"] for r in pair_reports if r["gap"] == gap]
        if vals:
            per_gap[str(gap)] = {
                "median_scale": float(np.median(vals)),
                "num_pairs": len(vals),
            }

    report = {
        "method": "moge_metric_depth_rgb_correspondence_omega_pose_scale",
        "camera_id": args.camera_id,
        "gaps": args.gaps,
        "num_pairs": len(pair_reports),
        "global_scale": global_scale,
        "global_log_mad": global_log_mad,
        "global_pair_inliers": (
            int(global_keep.sum()) if len(all_pair_scales) else 0
        ),
        "per_gap": per_gap,
        "pairs": pair_reports,
        "uses_gt_geometry": False,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    print("\n================ RESULT ================")
    print(f"global scale : {global_scale}")
    print(f"log MAD      : {global_log_mad}")
    print(f"pairs        : {len(pair_reports)}")
    print("per gap      :", per_gap)
    print("========================================")


if __name__ == "__main__":
    main()
