import numpy as np

from tools.export_ufo_pose_override import (
    apply_similarity,
    orientation_aware_similarity,
    umeyama_similarity,
)


def test_umeyama_similarity_recovers_transform():
    source = np.asarray([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=float)
    rotation = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    scale = 2.5
    translation = np.asarray([3, -4, 5], dtype=float)
    target = scale * (rotation @ source.T).T + translation
    actual_scale, actual_rotation, actual_translation = umeyama_similarity(source, target)
    np.testing.assert_allclose(actual_scale, scale)
    np.testing.assert_allclose(actual_rotation, rotation, atol=1e-12)
    np.testing.assert_allclose(actual_translation, translation, atol=1e-12)

    c2w = np.tile(np.eye(4), (len(source), 1, 1))
    c2w[:, :3, 3] = source
    aligned = apply_similarity(c2w, actual_scale, actual_rotation, actual_translation)
    np.testing.assert_allclose(aligned[:, :3, 3], target, atol=1e-12)


def test_orientation_aware_similarity_handles_straight_trajectory():
    source = np.tile(np.eye(4), (4, 1, 1))
    source[:, 0, 3] = np.arange(4)
    camera_yaw = np.deg2rad([0.0, 1.0, -1.0, 0.5])
    source[:, 0, 0] = np.cos(camera_yaw)
    source[:, 0, 1] = -np.sin(camera_yaw)
    source[:, 1, 0] = np.sin(camera_yaw)
    source[:, 1, 1] = np.cos(camera_yaw)

    angle = np.deg2rad(70)
    global_rotation = np.asarray(
        [
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)],
        ],
        dtype=float,
    )
    scale = 7.5
    translation = np.asarray([2.0, -3.0, 4.0])
    target = apply_similarity(source, scale, global_rotation, translation)

    actual_scale, actual_rotation, actual_translation = orientation_aware_similarity(
        source, target
    )
    np.testing.assert_allclose(actual_scale, scale, atol=1e-12)
    np.testing.assert_allclose(actual_rotation, global_rotation, atol=1e-12)
    np.testing.assert_allclose(actual_translation, translation, atol=1e-12)
