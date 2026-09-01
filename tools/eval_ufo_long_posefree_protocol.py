#!/usr/bin/env python3

import copy
import json
import logging
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import inference as ufo_inference
from ufo.dataset.constants import MEAN, STD
from ufo.dataset.data_utils import prepare_inputs_and_targets, to_batch_tensor
from ufo.dataset.pose_override import PoseOverrideStore
from ufo.utils.config import merge_config_and_args


LOGGER = logging.getLogger("UFO.long_eval")


def build_args():
    parser = ufo_inference.get_args_parser()

    parser.add_argument(
        "--start_indices",
        type=int,
        nargs="+",
        default=[0, 20, 40, 60, 80, 100, 120, 140, 160, 178],
    )
    parser.add_argument(
        "--pose_override_sequence_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--intrinsics_override_sequence_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--video_name",
        default="storm_velocity_20s_3cam.mp4",
    )

    args = merge_config_and_args(parser, config_path=None)
    args = merge_config_and_args(parser, config_path=args.config)
    args = ufo_inference.add_missing_config_values(args, args.config)

    # Exactly the previous Pose-Free long-sequence protocol:
    # each 2s window renders all 20 source timesteps,
    # then windows are deduplicated into frames 0..197.
    args.full_window_targets = True

    return args


def frame_ids(target):
    T = target["target_image"].shape[1]
    ids = target["target_frame_idx"][0].reshape(T, -1)

    if not torch.all(ids == ids[:, :1]):
        raise RuntimeError("Different cameras have inconsistent frame ids")

    return ids[:, 0].detach().cpu().numpy().astype(int)


def select_t(target, keep):
    total = target["target_image"].shape[1]
    index = torch.as_tensor(
        keep,
        dtype=torch.long,
        device=target["target_image"].device,
    )

    out = {}
    for key, value in target.items():
        if (
            isinstance(value, torch.Tensor)
            and value.ndim >= 2
            and value.shape[1] == total
        ):
            out[key] = value.index_select(1, index)
        else:
            out[key] = value

    return out


def select_pred(pred, total, keep):
    index = torch.as_tensor(
        keep,
        dtype=torch.long,
        device=pred["render_results"][
            pred["render_results"]["rgb_key"]
        ].device,
    )

    out = dict(pred)
    render = dict(pred["render_results"])

    for key, value in render.items():
        if (
            isinstance(value, torch.Tensor)
            and value.ndim >= 2
            and value.shape[1] == total
        ):
            render[key] = value.index_select(1, index)

    out["render_results"] = render
    return out


def video_frames(pred, device):
    mean = torch.tensor([[MEAN]], device=device)
    std = torch.tensor([[STD]], device=device)

    render = pred["render_results"]
    rgb = render[render["rgb_key"]][0]

    rgb = (rgb * std + mean).clamp(0, 1)

    for timestep in rgb:
        frame = torch.cat(
            [view for view in timestep],
            dim=1,
        )
        yield (
            frame.detach().cpu().float().numpy() * 255
        ).astype(np.uint8)


def make_metric_dataset_args(model_args):
    """
    Important:
    This is a completely separate Dataset instance.

    GT depth / dynamic mask / sky mask are loaded ONLY here for scoring.
    They never enter run_inference() or the UFO model.
    """
    args = copy.deepcopy(model_args)

    args.load_depth = True
    args.load_dynamic_mask = True
    args.skip_sky_mask = False

    args.load_flow = False
    args.load_ground = False

    args.full_window_targets = True

    return args


def get_metric_target(dataset, args, start_idx, device):
    dataset_index = 0 if args.annotation_file else args.scene_id

    raw = dataset.__getitem__(
        dataset_index,
        start_idx,
        return_all=True,
    )
    raw = to_batch_tensor(raw)

    chunks = prepare_inputs_and_targets(
        raw,
        device,
        timespan=args.timespan,
        from_list=True,
        args=args,
    )

    # We ONLY retain the GT target side.
    _, target = ufo_inference.concatenate_chunk_targets(chunks)

    return target


def update_sequence_override(dataset, args, start_idx):
    if (
        args.pose_override_mode != "none"
        and args.pose_override_sequence_dir
    ):
        root = (
            Path(args.pose_override_sequence_dir)
            / f"start_{start_idx:03d}"
        )
        dataset.pose_override_store = PoseOverrideStore(root)

    if (
        args.intrinsics_override_mode != "none"
        and args.intrinsics_override_sequence_dir
    ):
        root = (
            Path(args.intrinsics_override_sequence_dir)
            / f"start_{start_idx:03d}"
        )
        dataset.intrinsics_override_store = PoseOverrideStore(root)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    args = build_args()

    # ---------------------------------------------------------
    # Dataset A: MODEL INPUT
    # RGB + camera/K only.
    # ---------------------------------------------------------
    model_args = copy.deepcopy(args)

    model_args.load_depth = False
    model_args.load_dynamic_mask = False
    model_args.load_flow = False
    model_args.load_ground = False
    model_args.skip_sky_mask = True

    # ---------------------------------------------------------
    # Dataset B: METRICS ONLY
    # Contains GT labels but NEVER enters the model.
    # ---------------------------------------------------------
    metric_args = make_metric_dataset_args(args)

    device = torch.device(args.device)

    model = ufo_inference.build_model(
        model_args,
        device,
    )

    model_dataset = ufo_inference.build_dataset(
        model_args
    )
    metric_dataset = ufo_inference.build_dataset(
        metric_args
    )

    scene_frames = int(
        model_dataset.annotations[0]["num_timesteps"]
    )
    fps = float(
        model_dataset.annotations[0].get("fps", 10)
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_path = output_dir / args.video_name

    writer = imageio.get_writer(
        video_path,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )

    last_frame = -1
    mapping = []
    window_results = []

    # Exact legacy Pose-Free aggregation:
    legacy_sum = {}
    legacy_total_images = 0

    # Scientifically cleaner aggregation:
    valid_sum = {}
    valid_denominator = {}

    try:
        for start in args.start_indices:

            LOGGER.info("=" * 60)
            LOGGER.info("Window start=%d", start)

            # Same per-window camera override for both datasets.
            update_sequence_override(
                model_dataset,
                model_args,
                start,
            )
            update_sequence_override(
                metric_dataset,
                metric_args,
                start,
            )

            model_args.start_idx = start
            metric_args.start_idx = start

            # -------------------------------------------------
            # MODEL sees ONLY RGB + camera/K.
            # -------------------------------------------------
            pred, model_target, model_input, _ = (
                ufo_inference.run_inference(
                    model,
                    model_dataset,
                    model_args,
                    device,
                )
            )

            # -------------------------------------------------
            # Separate Dataset provides metric GT only.
            # Absolutely no GT metric tensor enters model.
            # -------------------------------------------------
            metric_target = get_metric_target(
                metric_dataset,
                metric_args,
                start,
                device,
            )

            ids_model = frame_ids(model_target)
            ids_metric = frame_ids(metric_target)

            if not np.array_equal(ids_model, ids_metric):
                raise RuntimeError(
                    f"Model/metric frame mismatch at start={start}"
                )

            keep = np.flatnonzero(
                ids_model > last_frame
            )

            if len(keep) == 0:
                continue

            selected_ids = ids_model[keep]

            total = model_target["target_image"].shape[1]

            selected_pred = select_pred(
                pred,
                total,
                keep,
            )

            selected_metric_target = select_t(
                metric_target,
                keep,
            )

            # -------------------------------------------------
            # Save continuous 20s render.
            # -------------------------------------------------
            for source_id, frame in zip(
                selected_ids,
                video_frames(selected_pred, device),
            ):
                writer.append_data(frame)

                mapping.append({
                    "video_frame": len(mapping),
                    "source_frame": int(source_id),
                    "window_start": int(start),
                })

            # -------------------------------------------------
            # EXACT SAME compute_metrics() as Pose-Free eval.
            # -------------------------------------------------
            metrics = ufo_inference.compute_metrics(
                selected_pred,
                selected_metric_target,
                model_input,
                device,
            )

            views = int(
                selected_metric_target[
                    "target_image"
                ].shape[2]
            )

            n_images = (
                len(selected_ids) * views
            )

            # ---------------------------------------------
            # 1. Legacy Pose-Free aggregation.
            # Needed for direct comparison with old results.
            # ---------------------------------------------
            for key, value in metrics.items():
                if np.isfinite(value):
                    legacy_sum[key] = (
                        legacy_sum.get(key, 0.0)
                        + value * n_images
                    )

            legacy_total_images += n_images

            # ---------------------------------------------
            # 2. Valid-only weighted aggregation.
            # Avoids NaN windows contaminating denominator.
            # ---------------------------------------------
            for key, value in metrics.items():
                if np.isfinite(value):
                    valid_sum[key] = (
                        valid_sum.get(key, 0.0)
                        + value * n_images
                    )
                    valid_denominator[key] = (
                        valid_denominator.get(key, 0)
                        + n_images
                    )

            window_results.append({
                "start_idx": int(start),
                "first_frame": int(selected_ids[0]),
                "last_frame": int(selected_ids[-1]),
                "num_frames": int(len(selected_ids)),
                "metrics": metrics,
            })

            print(
                f"[start={start:03d}] "
                f"frames={selected_ids[0]}-{selected_ids[-1]} "
                f"PSNR={metrics['psnr']:.3f} "
                f"SSIM={metrics['ssim']:.4f} "
                f"Dynamic={metrics['dynamic_psnr']:.3f}"
            )

            last_frame = int(
                selected_ids[-1]
            )

            del (
                pred,
                model_target,
                metric_target,
                selected_pred,
                selected_metric_target,
                model_input,
            )

            torch.cuda.empty_cache()

    finally:
        writer.close()

    legacy_metrics = {
        key: value / legacy_total_images
        for key, value in legacy_sum.items()
    }

    valid_metrics = {
        key: valid_sum[key] / valid_denominator[key]
        for key in valid_sum
    }

    # Strict continuous-scene check.
    source_frames = [
        x["source_frame"]
        for x in mapping
    ]

    full_protocol_starts = [
        0, 20, 40, 60, 80,
        100, 120, 140, 160, 178,
    ]

    # Strict continuity check only for the official full198 protocol.
    if list(args.start_indices) == full_protocol_starts:
        if len(source_frames) != scene_frames:
            raise RuntimeError(
                f"Expected {scene_frames} frames, "
                f"got {len(source_frames)}"
            )

        if source_frames != list(range(scene_frames)):
            raise RuntimeError(
                "Rendered frames are not exactly 0..197"
            )

    summary = {
        "checkpoint": str(
            Path(args.checkpoint).resolve()
        ),
        "scene_frames": scene_frames,
        "rendered_frames": len(mapping),
        "fps": fps,
        "duration_seconds": len(mapping) / fps,

        "window_starts": args.start_indices,

        "model_input": [
            "RGB",
            "camera pose",
            "camera intrinsics",
        ],

        "evaluation_only_gt": [
            "RGB",
            "dynamic mask",
            "depth",
            "sky mask",
        ],

        "window_state_policy":
            "reset_between_2s_windows",

        # THIS is the number to compare against
        # previous Pose-Free long-sequence results.
        "posefree_legacy_metrics": legacy_metrics,

        # Also report cleaner version.
        "valid_window_weighted_metrics":
            valid_metrics,

        "windows": window_results,

        "video": str(video_path.resolve()),
    }

    metrics_path = output_dir / "metrics.json"
    mapping_path = output_dir / "frame_mapping.json"

    metrics_path.write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    mapping_path.write_text(
        json.dumps(mapping, indent=2) + "\n"
    )

    print("\n" + "=" * 72)
    print("FULL 198-FRAME POSE-FREE-COMPATIBLE EVALUATION")
    print("=" * 72)

    for key, value in legacy_metrics.items():
        if "psnr" in key:
            print(
                f"{key:28s}: {value:.3f} dB"
            )
        elif "ssim" in key:
            print(
                f"{key:28s}: {value:.4f}"
            )
        elif "rmse" in key:
            print(
                f"{key:28s}: {value:.3f} m"
            )
        else:
            print(
                f"{key:28s}: {value}"
            )

    print("=" * 72)
    print("video  :", video_path)
    print("metrics:", metrics_path)


if __name__ == "__main__":
    main()
