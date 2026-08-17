import json
from collections import OrderedDict

from ufo.dataset.dataset import UFODataset


def make_dataset(tmp_path, object_translation):
    instance_dir = tmp_path / "datasets/waymo/validation/155/instances"
    instance_dir.mkdir(parents=True)
    instances = {
        "0": {
            "frame_annotations": {
                "frame_idx": [0],
                "obj_to_world": [[
                    [1, 0, 0, object_translation[0]],
                    [0, 1, 0, object_translation[1]],
                    [0, 0, 1, object_translation[2]],
                    [0, 0, 0, 1],
                ]],
            }
        }
    }
    (instance_dir / "instances_info.json").write_text(json.dumps(instances))
    (instance_dir / "frame_instances.json").write_text(json.dumps({"0": [0]}))
    dataset = UFODataset.__new__(UFODataset)
    dataset.data_root = str(tmp_path)
    dataset._instance_cache = OrderedDict()
    dataset._instance_scene_manifest = {
        "expected-scene": "datasets/waymo/validation/155/instances"
    }
    return dataset


def scene():
    return {
        "scene_name": "expected-scene",
        "dataset": "waymo",
        "ego_pose": [[[1, 0, 0, 10], [0, 1, 0, 20], [0, 0, 1, 30], [0, 0, 0, 1]]],
    }


def test_manifest_resolves_scene_name_instead_of_dataset_index(tmp_path):
    dataset = make_dataset(tmp_path, [12, 20, 30])
    loaded = dataset._with_instances(scene())
    assert loaded["instances_info"]["0"]["frame_annotations"]["obj_to_world"][0][0][3] == 12


def test_manifest_fails_fast_on_misaligned_instance_coordinates(tmp_path):
    dataset = make_dataset(tmp_path, [5000, -8000, 30])
    try:
        dataset._with_instances(scene())
    except ValueError as error:
        assert "instance/scene alignment failed" in str(error)
    else:
        raise AssertionError("misaligned instance coordinates must fail")
