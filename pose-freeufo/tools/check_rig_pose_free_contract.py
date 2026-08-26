#!/usr/bin/env python3
"""Validate that a rig-local pose export contains no GT camera trajectory leakage."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from argparse import Namespace
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ufo.dataset.dataset import UFODataset
from ufo.dataset.constants import DATASET_DICT, DATASETS
from ufo.paper_contract import relative_se3


FORBIDDEN_KEYS = {
    "camera_to_world", "ego_pose", "ego_to_world", "gt_c2w_opencv",
    "gt_camera_to_world", "sim3_scale", "sim3_rotation", "sim3_translation",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--pose-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--annotation-file", type=Path, required=True)
    parser.add_argument("--scene-index", type=int, required=True)
    return parser.parse_args()


def collect_keys(value):
    keys = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(key)
            keys.update(collect_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(collect_keys(child))
    return keys


def main():
    args = parse_args()
    windows = []
    all_baselines = {}
    for manifest_path in args.manifests:
        manifest = json.loads(manifest_path.read_text())
        leaked = sorted(collect_keys(manifest) & FORBIDDEN_KEYS)
        if leaked:
            raise ValueError(f"forbidden manifest keys in {manifest_path}: {leaked}")
        if manifest.get("pose_contract", {}).get("name") != "rig_pose_free_v1":
            raise ValueError(f"missing rig_pose_free_v1 contract: {manifest_path}")
        npz_path = (
            args.pose_root / f"start_{manifest['start_index']:03d}"
            / manifest["scene_name"] / "omega_pose_override.npz"
        )
        with np.load(npz_path, allow_pickle=False) as payload:
            keys = set(payload.files)
            leaked_npz = sorted(keys & FORBIDDEN_KEYS)
            if leaked_npz or any("aligned" in key or "sim3" in key for key in keys):
                raise ValueError(f"forbidden NPZ pose fields in {npz_path}: {leaked_npz}")
            if str(payload["coordinate_frame"].item()) != "rig_local_metric":
                raise ValueError(f"wrong coordinate frame in {npz_path}")
            if str(payload["metric_scale_source"].item()) != "fixed_camera_to_ego_baselines":
                raise ValueError(f"wrong metric source in {npz_path}")
            frame_ids = payload["frame_ids"].astype(int)
            camera_ids = payload["camera_ids"].astype(str)
            local = payload["omega_c2w_rig_local"].astype(np.float64)
            scale = float(payload["rig_metric_scale"])
        first_frame = int(frame_ids.min())
        front = np.flatnonzero((frame_ids == first_frame) & (camera_ids == "0"))
        identity_error = float(np.abs(local[int(front[0])] - np.eye(4)).max())
        if identity_error > 1e-5:
            raise ValueError(f"front0 gauge is not identity: {identity_error}")
        baseline_checks = {}
        rig = manifest["rig_camera_to_ego"]
        cameras = sorted(set(camera_ids.tolist()))
        for left, right in combinations(cameras, 2):
            real = float(np.linalg.norm(
                np.asarray(rig[left])[:3, 3] - np.asarray(rig[right])[:3, 3]
            ))
            predicted = []
            for frame_id in sorted(set(frame_ids.tolist())):
                li = int(np.flatnonzero((frame_ids == frame_id) & (camera_ids == left))[0])
                ri = int(np.flatnonzero((frame_ids == frame_id) & (camera_ids == right))[0])
                predicted.append(float(np.linalg.norm(
                    local[li, :3, 3] - local[ri, :3, 3]
                )))
            baseline_checks[f"{left}-{right}"] = {
                "real_m": real,
                "predicted_median_m": float(np.median(predicted)),
                "median_absolute_error_m": float(np.median(np.abs(np.asarray(predicted) - real))),
            }
            aggregate = all_baselines.setdefault(
                f"{left}-{right}", {"real_m": real, "predicted_m": []}
            )
            aggregate["predicted_m"].extend(predicted)
        windows.append({
            "start_index": manifest["start_index"],
            "manifest": str(manifest_path.resolve()),
            "pose_override": str(npz_path.resolve()),
            "rig_metric_scale": scale,
            "front0_identity_max_abs_error": identity_error,
            "baselines": baseline_checks,
        })
    scales = np.asarray([item["rig_metric_scale"] for item in windows])
    aggregate_baselines = {}
    for pair, values in all_baselines.items():
        predicted = np.asarray(values["predicted_m"])
        errors = np.abs(predicted - values["real_m"])
        aggregate_baselines[pair] = {
            "real_m": values["real_m"],
            "predicted_median_m": float(np.median(predicted)),
            "absolute_error_median": float(np.median(errors)),
            "absolute_error_m_mean": float(errors.mean()),
            "observations": len(predicted),
        }
    config = json.loads(args.config.read_text())
    coordinate_mode = config.get("pose_free_coordinate_mode", "identity")
    if coordinate_mode not in ("identity", "recurrent"):
        raise ValueError(f"invalid pose_free_coordinate_mode={coordinate_mode!r}")
    config["pose_override_dir"] = str(args.pose_root / f"start_{windows[0]['start_index']:03d}")
    dataset = UFODataset(
        data_root=config["data_root"],
        annotation_txt_file_list=str(args.annotation_file),
        subset_indices=[args.scene_index],
        target_size=tuple(config["input_size"]),
        num_context_timesteps=config["num_context_timesteps"],
        num_target_timesteps=config["num_target_timesteps"],
        num_max_cams=config["num_max_cameras"],
        timespan=config["timespan"],
        equispaced=True,
        load_depth=config["load_depth"],
        load_flow=False,
        load_dynamic_mask=config["load_dynamic_mask"],
        load_ground_label=config["load_ground"],
        skip_sky_mask=config["skip_sky_mask"],
        num_target_chunks=config["num_target_chunks"],
        args=Namespace(**config),
    )
    removed = []
    for forbidden in ("camera_to_world", "ego_pose", "ego_to_world"):
        if dataset.annotations[0].pop(forbidden, None) is not None:
            removed.append(forbidden)
    chunks = dataset.__getitem__(0, windows[0]["start_index"], return_all=True)
    runtime_check = {
        "passed": True,
        "removed_before_dataset_getitem": removed,
        "chunks_checked": len(chunks),
        "coordinate_mode": coordinate_mode,
        "object_instance_ids_zero": True,
    }
    for chunk in chunks:
        for role in ("context", "target"):
            if int(chunk[role]["instances_id"].sum()) != 0:
                raise ValueError(f"{role} contains GT-world object instances")

    if coordinate_mode == "identity":
        for chunk in chunks:
            for role in ("context", "target"):
                payload = chunk[role]
                if not torch.equal(payload["camtoworld"], payload["camtoworld_global"]):
                    raise ValueError(f"legacy {role} local/global camera frames differ")
        runtime_check["local_global_camera_matrices_identical"] = True
    else:
        scene_name = dataset.annotations[0]["scene_name"]
        dataset_name = dataset.annotations[0]["dataset"]
        cameras = DATASET_DICT[dataset_name]["camera_list"][config["num_max_cameras"]]
        ref_camera = DATASET_DICT[dataset_name]["ref_camera"]
        canonical_to_flu = np.asarray(
            DATASETS[dataset_name]["canonical_to_flu"], dtype=np.float64
        )
        opencv_to_dataset = np.asarray(
            DATASETS[dataset_name]["opencv2dataset"], dtype=np.float64
        )
        global_frame = dataset.pose_override_store.frame_ids(scene_name)[-1]
        global_ref = dataset.pose_override_store.get(scene_name, global_frame, ref_camera)
        world_to_global = np.linalg.inv(global_ref)
        formula_max_error = 0.0
        local_global_max_differences = []
        scene_from_local_identity_errors = []
        for chunk in chunks:
            source_frame = int(chunk["context"]["frame_idx"][0].item())
            local_ref = dataset.pose_override_store.get(scene_name, source_frame, ref_camera)
            world_to_local = np.linalg.inv(local_ref)
            for role in ("context", "target"):
                payload = chunk[role]
                frame_ids = payload["frame_idx"].to(torch.int64).tolist()
                expected_local = []
                expected_global = []
                for frame_id in frame_ids:
                    for camera in cameras:
                        pose = dataset.pose_override_store.get(scene_name, frame_id, camera)
                        expected_local.append(
                            canonical_to_flu @ world_to_local @ pose @ opencv_to_dataset
                        )
                        expected_global.append(
                            canonical_to_flu @ world_to_global @ pose @ opencv_to_dataset
                        )
                expected_local = torch.from_numpy(np.stack(expected_local)).float()
                expected_global = torch.from_numpy(np.stack(expected_global)).float()
                formula_max_error = max(
                    formula_max_error,
                    float((payload["camtoworld"] - expected_local).abs().max()),
                    float((payload["camtoworld_global"] - expected_global).abs().max()),
                )
                local_global_max_differences.append(float(
                    (payload["camtoworld"] - payload["camtoworld_global"]).abs().max()
                ))
            scene_from_local = relative_se3(
                chunk["context"]["camtoworld_global"][0].float(),
                chunk["context"]["camtoworld"][0].float(),
            )
            scene_from_local_identity_errors.append(float(
                (scene_from_local - torch.eye(4)).abs().max()
            ))
        if formula_max_error > 1e-5:
            raise ValueError(f"runtime camera formula mismatch: {formula_max_error}")
        if not all(value > 1e-6 for value in local_global_max_differences):
            raise ValueError("P2.1 unexpectedly collapsed a local/global camera pair")
        if not all(value > 1e-6 for value in scene_from_local_identity_errors):
            raise ValueError("P2.1 scene_from_local unexpectedly became identity")
        runtime_check.update({
            "global_reference_policy": "last_frame_in_pose_override",
            "global_reference_frame": global_frame,
            "camera_formula_max_abs_error": formula_max_error,
            "local_global_max_abs_differences": local_global_max_differences,
            "scene_from_local_identity_max_abs_errors": scene_from_local_identity_errors,
            "local_global_camera_matrices_distinct": True,
        })

    report = {
        "contract": (
            "rig_pose_free_recurrent_v2"
            if coordinate_mode == "recurrent" else "rig_pose_free_v1"
        ),
        "passed": True,
        "forbidden_camera_pose_sources": sorted(FORBIDDEN_KEYS),
        "allowed_metric_source": "fixed camera_to_ego rig calibration",
        "num_windows": len(windows),
        "scale_mean": float(scales.mean()),
        "scale_std": float(scales.std()),
        "scale_coefficient_of_variation": float(scales.std() / scales.mean()),
        "aggregate_baselines": aggregate_baselines,
        "runtime_sanitized_annotation_check": runtime_check,
        "windows": windows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
