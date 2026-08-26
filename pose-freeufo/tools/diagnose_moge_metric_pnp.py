#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from diagnose_rgb_pose_scale import mutual_sift_matches


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--omega-npz", type=Path, required=True)
    p.add_argument("--moge-repo", type=Path, required=True)
    p.add_argument("--moge-model", type=Path, required=True)
    p.add_argument("--camera-id", default="0")
    p.add_argument("--gaps", type=int, nargs="+", default=[1, 2, 4, 6])
    p.add_argument("--ratio", type=float, default=0.75)
    p.add_argument("--resolution-level", type=int, default=9)
    p.add_argument("--pnp-reproj", type=float, default=3.0)
    p.add_argument("--min-matches", type=int, default=20)
    p.add_argument("--min-inliers", type=int, default=12)
    p.add_argument("--max-depth", type=float, default=120.0)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def rotation_angle_deg(R):
    value = (np.trace(R) - 1.0) / 2.0
    value = np.clip(value, -1.0, 1.0)
    return float(np.degrees(np.arccos(value)))


def make_T(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).reshape(3)
    return T


def normalized_K_to_pixels(K, width, height):
    S = np.array([
        [width, 0.0, 0.0],
        [0.0, height, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    return S @ K


def main():
    args = parse_args()

    manifest = json.loads(args.manifest.read_text())
    entries = {
        (int(e["frame_id"]), str(e["camera_id"])): e
        for e in manifest["images"]
    }

    with np.load(args.omega_npz, allow_pickle=False) as x:
        omega_frames = x["frame_ids"].astype(int)
        omega_cams = x["camera_ids"].astype(str)
        omega_c2w = x["omega_c2w_raw"].astype(np.float64)

    omega_lookup = {
        (int(f), str(c)): i
        for i, (f, c) in enumerate(zip(omega_frames, omega_cams))
    }

    frames = sorted(
        f for (f, c) in entries
        if c == args.camera_id and (f, c) in omega_lookup
    )

    sys.path.insert(0, str(args.moge_repo))
    from moge.model.v2 import MoGeModel

    device = torch.device("cuda")
    print("[MoGe] loading model", flush=True)
    model = MoGeModel.from_pretrained(args.moge_model).eval().to(device)

    cache = {}

    def get_moge(frame):
        key = (frame, args.camera_id)
        if key in cache:
            return cache[key]

        path = Path(entries[key]["path"])
        rgb_np = np.asarray(Image.open(path).convert("RGB")).copy()
        H, W = rgb_np.shape[:2]

        rgb = (
            torch.from_numpy(rgb_np)
            .float()
            .permute(2, 0, 1)
            .to(device)
            / 255.0
        )

        with torch.inference_mode():
            out = model.infer(
                rgb,
                resolution_level=args.resolution_level,
                use_fp16=True,
                apply_mask=False,
            )

        points = out["points"].float().cpu().numpy()
        mask = out["mask"].cpu().numpy().astype(bool)
        K_norm = out["intrinsics"].float().cpu().numpy()
        K_pix = normalized_K_to_pixels(K_norm, W, H)

        gray = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY)

        result = {
            "points": points,
            "mask": mask,
            "K": K_pix,
            "gray": gray,
            "height": H,
            "width": W,
        }
        cache[key] = result
        return result

    def solve(src_frame, dst_frame, p_src, p_dst):
        src = get_moge(src_frame)
        dst = get_moge(dst_frame)

        X = []
        uv = []

        H, W = src["height"], src["width"]

        for a, b in zip(p_src, p_dst):
            x = int(round(a[0]))
            y = int(round(a[1]))

            if x < 0 or x >= W or y < 0 or y >= H:
                continue

            if not src["mask"][y, x]:
                continue

            P = src["points"][y, x].astype(np.float64)

            if not np.all(np.isfinite(P)):
                continue
            if P[2] <= 0.2 or P[2] > args.max_depth:
                continue

            # Target MoGe validity is useful for rejecting sky / invalid regions.
            tx = int(round(b[0]))
            ty = int(round(b[1]))
            if (
                tx < 0 or tx >= dst["width"]
                or ty < 0 or ty >= dst["height"]
            ):
                continue
            if not dst["mask"][ty, tx]:
                continue

            X.append(P)
            uv.append(b)

        if len(X) < args.min_matches:
            return None

        X = np.asarray(X, dtype=np.float64)
        uv = np.asarray(uv, dtype=np.float64)

        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            X,
            uv,
            dst["K"],
            None,
            iterationsCount=3000,
            reprojectionError=args.pnp_reproj,
            confidence=0.999,
            flags=cv2.SOLVEPNP_EPNP,
        )

        if not ok or inliers is None:
            return None

        inliers = inliers.reshape(-1)

        if len(inliers) < args.min_inliers:
            return None

        Xi = X[inliers]
        uvi = uv[inliers]

        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                Xi,
                uvi,
                dst["K"],
                None,
                rvec,
                tvec,
            )
        except cv2.error:
            pass

        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3)

        proj, _ = cv2.projectPoints(
            Xi,
            rvec,
            tvec,
            dst["K"],
            None,
        )
        proj = proj.reshape(-1, 2)

        reproj = np.linalg.norm(proj - uvi, axis=1)

        return {
            "T": make_T(R, t),
            "R": R,
            "t": t,
            "matches_3d2d": int(len(X)),
            "inliers": int(len(inliers)),
            "inlier_ratio": float(len(inliers) / len(X)),
            "median_reprojection_px": float(np.median(reproj)),
        }

    reports = []

    for gap in args.gaps:
        for f0 in frames:
            f1 = f0 + gap
            if f1 not in frames:
                continue

            m0 = get_moge(f0)
            m1 = get_moge(f1)

            p0, p1 = mutual_sift_matches(
                m0["gray"], m1["gray"], args.ratio
            )

            if len(p0) < args.min_matches:
                continue

            forward = solve(f0, f1, p0, p1)
            reverse = solve(f1, f0, p1, p0)

            if forward is None or reverse is None:
                continue

            T_fwd = forward["T"]
            T_rev = reverse["T"]

            cycle = T_rev @ T_fwd
            cycle_rot = rotation_angle_deg(cycle[:3, :3])
            cycle_trans = float(np.linalg.norm(cycle[:3, 3]))

            # Omega is comparison/initialization diagnostic only.
            # It is NOT used to solve the metric PnP pose.
            i0 = omega_lookup[(f0, args.camera_id)]
            i1 = omega_lookup[(f1, args.camera_id)]

            C0 = omega_c2w[i0]
            C1 = omega_c2w[i1]

            T_omega = np.linalg.inv(C1) @ C0

            R_omega = T_omega[:3, :3]
            t_omega = T_omega[:3, 3]

            R_pnp = T_fwd[:3, :3]
            t_pnp = T_fwd[:3, 3]

            rot_error = rotation_angle_deg(R_pnp @ R_omega.T)

            npnp = float(np.linalg.norm(t_pnp))
            nomega = float(np.linalg.norm(t_omega))

            if npnp > 1e-8 and nomega > 1e-8:
                direction_cos = float(
                    np.dot(t_pnp, t_omega) / (npnp * nomega)
                )
                equivalent_scale = float(npnp / nomega)
            else:
                direction_cos = None
                equivalent_scale = None

            row = {
                "frame0": int(f0),
                "frame1": int(f1),
                "gap": int(gap),
                "sift_mutual": int(len(p0)),
                "pnp_translation_m": npnp,
                "pnp_rotation_deg": rotation_angle_deg(R_pnp),
                "pnp_inliers": forward["inliers"],
                "pnp_inlier_ratio": forward["inlier_ratio"],
                "median_reprojection_px":
                    forward["median_reprojection_px"],
                "reverse_translation_m":
                    float(np.linalg.norm(T_rev[:3, 3])),
                "cycle_rotation_deg": cycle_rot,
                "cycle_translation_m": cycle_trans,
                "omega_rotation_difference_deg": rot_error,
                "omega_translation_direction_cos": direction_cos,
                "omega_equivalent_scale": equivalent_scale,
            }

            reports.append(row)

            cos_text = (
                f"{direction_cos:.3f}"
                if direction_cos is not None else "nan"
            )

            print(
                f"{f0:03d}->{f1:03d} gap={gap} | "
                f"t={npnp:6.3f}m | "
                f"inliers={forward['inliers']:3d} "
                f"({forward['inlier_ratio']:.2f}) | "
                f"reproj={forward['median_reprojection_px']:.2f}px | "
                f"cycle={cycle_trans:.3f}m/"
                f"{cycle_rot:.2f}deg | "
                f"dR_omega={rot_error:.2f}deg | "
                f"dir={cos_text}",
                flush=True,
            )

    per_gap = {}

    for gap in args.gaps:
        rows = [r for r in reports if r["gap"] == gap]
        if not rows:
            continue

        def med(key):
            vals = [
                r[key] for r in rows
                if r[key] is not None and np.isfinite(r[key])
            ]
            return float(np.median(vals)) if vals else None

        per_gap[str(gap)] = {
            "num_pairs": len(rows),
            "median_translation_m": med("pnp_translation_m"),
            "median_inlier_ratio": med("pnp_inlier_ratio"),
            "median_reprojection_px": med("median_reprojection_px"),
            "median_cycle_translation_m": med("cycle_translation_m"),
            "median_cycle_rotation_deg": med("cycle_rotation_deg"),
            "median_omega_rotation_difference_deg":
                med("omega_rotation_difference_deg"),
            "median_omega_translation_direction_cos":
                med("omega_translation_direction_cos"),
            "median_omega_equivalent_scale":
                med("omega_equivalent_scale"),
        }

    report = {
        "method": "moge2_metric_pointmap_rgb_pnp",
        "camera_id": args.camera_id,
        "gaps": args.gaps,
        "num_pairs": len(reports),
        "per_gap": per_gap,
        "pairs": reports,
        "uses_gt_pose": False,
        "uses_gt_intrinsics": False,
        "uses_gt_depth": False,
        "uses_camera_to_ego": False,
        "omega_used_for_solving": False,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    print("\n================ RESULT ================")
    print("method    : MoGe-2 metric point map + PnP")
    print("pairs     :", len(reports))
    print("per gap   :", json.dumps(per_gap, indent=2))
    print("GT geometry: NOT USED")
    print("========================================")


if __name__ == "__main__":
    main()
