#!/usr/bin/env python3
"""Export the exact RGB/pose sample selected by UFODataset."""

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ufo.dataset.constants import DATASET_DICT, DATASETS
from ufo.dataset.dataset import UFODataset


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--annotation-file", type=Path, required=True)
    parser.add_argument("--scene-index", type=int, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def image_path(data_root, dataset_name, relative_path):
    if dataset_name in ("waymo", "nuscenes", "argoverse2", "argoverse"):
        relative_path = relative_path.replace("images", "images_4")
        relative_path = relative_path.replace("sweeps", "sweeps_4")
        relative_path = relative_path.replace("samples", "samples_4")
    return data_root / "datasets" / dataset_name / relative_path


def main():
    cli = parse_args()
    config = json.loads(cli.config.read_text())
    config.update({
        "pose_override_mode": "none",
        "pose_override_dir": None,
        "num_bbox": config.get("num_bbox", 32),
    })
    dataset_args = Namespace(**config)
    dataset = UFODataset(
        data_root=str(cli.data_root),
        annotation_txt_file_list=str(cli.annotation_file),
        subset_indices=[cli.scene_index],
        target_size=tuple(config["input_size"]),
        num_context_timesteps=config["num_context_timesteps"],
        num_target_timesteps=config["num_target_timesteps"],
        num_max_cams=config["num_max_cameras"],
        timespan=config["timespan"],
        equispaced=True,
        load_depth=False,
        load_flow=False,
        load_dynamic_mask=False,
        load_ground_label=False,
        skip_sky_mask=True,
        num_target_chunks=config["num_target_chunks"],
        args=dataset_args,
    )
    chunks = dataset.__getitem__(0, cli.start_index, return_all=True)
    scene_json = dataset.annotations[0]
    dataset_name = scene_json["dataset"]
    cameras = DATASET_DICT[dataset_name]["camera_list"][config["num_max_cameras"]]
    opencv_to_dataset = np.asarray(DATASETS[dataset_name]["opencv2dataset"], dtype=np.float64)

    entries = []
    seen = set()
    for chunk_index, chunk in enumerate(chunks):
        for role in ("context", "target"):
            frame_ids = chunk[role]["frame_idx"].reshape(-1).tolist()
            for frame_id in frame_ids:
                frame_id = int(frame_id)
                for camera_id in cameras:
                    key = (frame_id, camera_id)
                    if key in seen:
                        raise ValueError(f"duplicate manifest image {key}")
                    seen.add(key)
                    relative_path = scene_json["relative_image_path"][camera_id][frame_id]
                    gt_native = np.asarray(
                        scene_json["camera_to_world"][camera_id][frame_id], dtype=np.float64
                    )
                    entries.append({
                        "frame_id": frame_id,
                        "camera_id": camera_id,
                        "role": role,
                        "chunk_index": chunk_index,
                        "path": str(image_path(cli.data_root, dataset_name, relative_path).resolve()),
                        "gt_camera_to_world": gt_native.tolist(),
                        "gt_c2w_opencv": (gt_native @ opencv_to_dataset).tolist(),
                    })

    manifest = {
        "schema_version": 1,
        "scene_index": cli.scene_index,
        "scene_id": scene_json["scene_id"],
        "scene_name": scene_json["scene_name"],
        "dataset": dataset_name,
        "start_index": cli.start_index,
        "camera_ids": cameras,
        "opencv_to_dataset": opencv_to_dataset.tolist(),
        "config": str(cli.config.resolve()),
        "annotation_file": str(cli.annotation_file.resolve()),
        "images": entries,
    }
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "output": str(cli.output),
        "scene_name": manifest["scene_name"],
        "context_images": sum(item["role"] == "context" for item in entries),
        "target_images": sum(item["role"] == "target" for item in entries),
    }, indent=2))


if __name__ == "__main__":
    main()
