"""Load per-scene camera poses exported by VGGT-Omega."""

from pathlib import Path

import numpy as np


class PoseOverrideStore:
    def __init__(self, root):
        self.root = Path(root)
        self._scene_name = None
        self._poses = None

    def _load(self, scene_name):
        path = self.root / scene_name / "omega_pose_override.npz"
        if not path.is_file():
            raise FileNotFoundError(f"pose override not found: {path}")
        with np.load(path, allow_pickle=False) as payload:
            payload_scene = str(payload["scene_name"].item())
            if payload_scene != scene_name:
                raise ValueError(
                    f"pose override scene mismatch: {payload_scene!r} != {scene_name!r}"
                )
            frame_ids = payload["frame_ids"].astype(np.int64)
            camera_ids = payload["camera_ids"].astype(str)
            poses = payload["omega_camera_to_world_aligned"].astype(np.float64)
        if poses.shape != (len(frame_ids), 4, 4):
            raise ValueError(f"invalid pose override shape {poses.shape} in {path}")
        mapping = {}
        for frame_id, camera_id, pose in zip(frame_ids, camera_ids, poses):
            key = (int(frame_id), str(camera_id))
            if key in mapping:
                raise ValueError(f"duplicate pose override entry {key} in {path}")
            mapping[key] = pose
        self._scene_name = scene_name
        self._poses = mapping

    def get(self, scene_name, frame_id, camera_id):
        if self._scene_name != scene_name:
            self._load(scene_name)
        key = (int(frame_id), str(camera_id))
        try:
            return self._poses[key]
        except KeyError as error:
            raise KeyError(
                f"pose override for scene={scene_name} frame={frame_id} camera={camera_id} is missing"
            ) from error
