#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ufo.dataset.constants import DATASET_DICT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--annotation-file", type=Path, required=True)
    p.add_argument("--scene-index", type=int, required=True)
    p.add_argument("--start-index", type=int, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    args = p.parse_args()

    cfg = json.loads(args.config.read_text())

    lines = [
        x.strip()
        for x in args.annotation_file.read_text().splitlines()
        if x.strip()
    ]

    annotation_path = Path(lines[args.scene_index])
    if not annotation_path.is_absolute():
        annotation_path = args.data_root / annotation_path

    scene = json.loads(annotation_path.read_text())

    fps = float(scene["fps"])
    frames_per_chunk = int(round(cfg["timespan"] * fps))
    total_frames = frames_per_chunk * cfg["num_target_chunks"]

    start = args.start_index
    end = start + total_frames

    if end > scene["num_timesteps"]:
        raise ValueError(
            f"invalid window [{start}, {end}) "
            f"for {scene['num_timesteps']} frames"
        )

    dataset_name = scene["dataset"]
    cameras = DATASET_DICT[dataset_name]["camera_list"][
        cfg["num_max_cameras"]
    ]
    ref_camera = DATASET_DICT[dataset_name]["ref_camera"]

    # scene_json camera_to_world 本身就是 UFO dataset 所期待的
    # native camera convention。
    ref_pose = np.asarray(
        scene["camera_to_world"][ref_camera][start],
        dtype=np.float64,
    )
    world_to_local = np.linalg.inv(ref_pose)

    frame_ids = []
    camera_ids = []
    poses = []
    intrinsics = []
    roles = []

    H, W = cfg["input_size"]

    for frame in range(start, end):
        for camera in cameras:
            raw_c2w = np.asarray(
                scene["camera_to_world"][camera][frame],
                dtype=np.float64,
            )

            # 与 Pose-Free override 使用相同 gauge：
            # 当前 window 第一帧 front camera = identity。
            local_c2w = world_to_local @ raw_c2w

            fx, fy, cx, cy = np.asarray(
                scene["normalized_intrinsics"][camera],
                dtype=np.float64,
            )

            K = np.array([
                [fx * W, 0.0, cx * W],
                [0.0, fy * H, cy * H],
                [0.0, 0.0, 1.0],
            ])

            frame_ids.append(frame)
            camera_ids.append(str(camera))
            poses.append(local_c2w)
            intrinsics.append(K)
            roles.append("gt_camera_ablation")

    scene_name = scene["scene_name"]

    out = (
        args.output_root
        / f"start_{start:03d}"
        / scene_name
        / "omega_pose_override.npz"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out,
        scene_name=np.asarray(scene_name),
        scope=np.asarray("all"),

        # 保持当前 Pose-Free dataset 完全相同的接口/代码路径。
        coordinate_frame=np.asarray("rig_local_metric"),
        metric_scale_source=np.asarray("gt_camera_ablation"),
        world_gauge=np.asarray("first_timestamp_front_camera"),

        frame_ids=np.asarray(frame_ids, dtype=np.int32),
        camera_ids=np.asarray(camera_ids),
        roles=np.asarray(roles),

        omega_camera_to_world_rig_local=np.asarray(
            poses, dtype=np.float32
        ),
        predicted_intrinsics_ufo=np.asarray(
            intrinsics, dtype=np.float32
        ),
    )

    identity_err = np.abs(
        np.asarray(poses)[
            (np.asarray(frame_ids) == start)
            & (np.asarray(camera_ids) == str(ref_camera))
        ][0]
        - np.eye(4)
    ).max()

    print(
        f"{start:03d}: frames={total_frames}, "
        f"poses={len(poses)}, "
        f"front_identity_err={identity_err:.3e}"
    )


if __name__ == "__main__":
    main()
