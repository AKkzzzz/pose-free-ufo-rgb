import numpy as np

from ufo.dataset.pose_override import PoseOverrideStore


def test_pose_override_store(tmp_path):
    scene = "scene-621"
    output = tmp_path / scene
    output.mkdir()
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = (1, 2, 3)
    np.savez_compressed(
        output / "omega_pose_override.npz",
        scene_name=np.asarray(scene),
        frame_ids=np.asarray([5]),
        camera_ids=np.asarray(["0"]),
        omega_camera_to_world_aligned=pose[None],
    )
    loaded = PoseOverrideStore(tmp_path).get(scene, 5, "0")
    np.testing.assert_allclose(loaded, pose)
