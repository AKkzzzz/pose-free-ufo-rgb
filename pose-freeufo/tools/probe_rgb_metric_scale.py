#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--omega-repo", type=Path, required=True)
    p.add_argument("--omega-checkpoint", type=Path, required=True)
    p.add_argument("--moge-repo", type=Path, required=True)
    p.add_argument("--moge-model", default="Ruicheng/moge-2-vitl")
    p.add_argument("--rig-oracle-npz", type=Path, default=None)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-images", type=int, default=0)
    p.add_argument("--resolution-level", type=int, default=9)
    return p.parse_args()


def robust_scale(metric_depth, omega_depth, omega_conf, moge_mask):
    metric_depth = np.asarray(metric_depth, np.float64)
    omega_depth = np.asarray(omega_depth, np.float64)
    omega_conf = np.asarray(omega_conf, np.float64)

    valid = (
        np.isfinite(metric_depth)
        & np.isfinite(omega_depth)
        & (metric_depth > 0.5)
        & (metric_depth < 120.0)
        & (omega_depth > 1e-5)
    )

    if moge_mask is not None:
        valid &= moge_mask.astype(bool)

    # 去掉边缘 5%，降低 resize / border 对尺度估计的影响
    h, w = valid.shape
    border = np.zeros_like(valid)
    y0, y1 = int(0.05 * h), int(0.95 * h)
    x0, x1 = int(0.05 * w), int(0.95 * w)
    border[y0:y1, x0:x1] = True
    valid &= border

    # 只保留 Omega confidence 较高的一半像素
    if valid.sum() < 100:
        raise RuntimeError(f"too few valid pixels before confidence filter: {valid.sum()}")

    conf_threshold = np.median(omega_conf[valid])
    valid &= omega_conf >= conf_threshold

    if valid.sum() < 100:
        raise RuntimeError(f"too few valid pixels after confidence filter: {valid.sum()}")

    # log-space 更稳定：
    # log(s) = median(log(D_metric) - log(D_omega))
    log_ratio = (
        np.log(metric_depth[valid])
        - np.log(omega_depth[valid])
    )

    log_scale = np.median(log_ratio)
    scale = float(np.exp(log_scale))

    mad = float(np.median(np.abs(log_ratio - log_scale)))

    return scale, mad, int(valid.sum())


def main():
    args = parse_args()

    manifest = json.loads(args.manifest.read_text())
    entries = manifest["images"]

    if args.max_images > 0:
        entries = entries[: args.max_images]

    image_paths = [item["path"] for item in entries]

    # ---------------------------------------------------
    # 1. VGGT-Omega: arbitrary-scale depth + confidence
    # ---------------------------------------------------
    sys.path.insert(0, str(args.omega_repo))

    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images

    device = torch.device("cuda")

    print(f"[Omega] loading {len(image_paths)} images", flush=True)

    omega_images = load_and_preprocess_images(
        image_paths,
        image_resolution=512
    ).to(device)

    omega = VGGTOmega().eval()
    state = torch.load(
        args.omega_checkpoint,
        map_location="cpu",
        weights_only=True
    )
    omega.load_state_dict(state)
    del state

    omega = omega.to(device)

    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=torch.bfloat16
    ):
        pred = omega(omega_images)

    omega_depth = pred["depth"][0].detach().float()
    omega_conf = pred["depth_conf"][0].detach().float()

    if omega_depth.ndim == 4 and omega_depth.shape[-1] == 1:
        omega_depth = omega_depth[..., 0]

    print("Omega depth:", tuple(omega_depth.shape))
    print("Omega conf :", tuple(omega_conf.shape))

    del omega, pred, omega_images
    torch.cuda.empty_cache()

    # ---------------------------------------------------
    # 2. MoGe-2: RGB -> metric depth
    # ---------------------------------------------------
    sys.path.insert(0, str(args.moge_repo))
    from moge.model.v2 import MoGeModel

    print("[MoGe-2] loading model", flush=True)

    moge = MoGeModel.from_pretrained(args.moge_model)
    moge = moge.to(device).eval()

    results = []

    for i, (entry, image_path) in enumerate(zip(entries, image_paths)):
        rgb_np = np.asarray(
            Image.open(image_path).convert("RGB")
        ).copy()

        h, w = rgb_np.shape[:2]

        rgb = torch.from_numpy(rgb_np).float().to(device)
        rgb = rgb.permute(2, 0, 1) / 255.0

        with torch.inference_mode():
            out = moge.infer(
                rgb,
                resolution_level=args.resolution_level,
                use_fp16=True,
                apply_mask=True,
            )

        metric_depth = out["depth"].detach().float().cpu().numpy()
        moge_mask = (
            out["mask"].detach().cpu().numpy()
            if "mask" in out else None
        )

        # Omega 是 416x624；映射回当前原图坐标
        od = F.interpolate(
            omega_depth[i][None, None],
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )[0, 0].cpu().numpy()

        oc = F.interpolate(
            omega_conf[i][None, None],
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )[0, 0].cpu().numpy()

        scale, log_mad, valid_pixels = robust_scale(
            metric_depth,
            od,
            oc,
            moge_mask,
        )

        row = {
            "index": i,
            "frame_id": int(entry["frame_id"]),
            "camera_id": str(entry["camera_id"]),
            "role": entry["role"],
            "scale": scale,
            "log_mad": log_mad,
            "valid_pixels": valid_pixels,
        }
        results.append(row)

        print(
            f"[{i+1:02d}/{len(entries)}] "
            f"frame={row['frame_id']:03d} "
            f"cam={row['camera_id']} "
            f"scale={scale:.5f} "
            f"pixels={valid_pixels}",
            flush=True,
        )

    # ---------------------------------------------------
    # 3. Aggregate: equal weight per image
    # ---------------------------------------------------
    scales = np.asarray([r["scale"] for r in results], np.float64)
    log_scales = np.log(scales)

    window_log_scale = np.median(log_scales)
    window_scale = float(np.exp(window_log_scale))

    image_log_mad = float(
        np.median(np.abs(log_scales - window_log_scale))
    )

    per_camera = {}
    for camera in sorted(set(r["camera_id"] for r in results)):
        values = np.asarray(
            [r["scale"] for r in results if r["camera_id"] == camera]
        )
        per_camera[camera] = {
            "median_scale": float(np.median(values)),
            "mean_scale": float(np.mean(values)),
            "std_scale": float(np.std(values)),
            "num_images": int(len(values)),
        }

    report = {
        "method": "rgb_only_moge2_to_omega_depth_scale",
        "moge_model": args.moge_model,
        "num_images": len(results),
        "rgb_metric_scale": window_scale,
        "image_log_mad": image_log_mad,
        "per_camera": per_camera,
        "images": results,
    }

    # rig scale 只作为 evaluation oracle，不参与估计
    if args.rig_oracle_npz is not None:
        with np.load(args.rig_oracle_npz, allow_pickle=False) as x:
            rig_scale = float(x["rig_metric_scale"])

        report["oracle_rig_scale"] = rig_scale
        report["relative_error_vs_rig"] = float(
            abs(window_scale / rig_scale - 1.0)
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    print("\n================ RESULT ================")
    print(f"RGB-only scale : {window_scale:.6f}")
    print(f"log MAD        : {image_log_mad:.6f}")

    if "oracle_rig_scale" in report:
        print(f"Rig oracle     : {report['oracle_rig_scale']:.6f}")
        print(
            f"Relative error : "
            f"{report['relative_error_vs_rig'] * 100:.2f}%"
        )

    print("========================================")


if __name__ == "__main__":
    main()
