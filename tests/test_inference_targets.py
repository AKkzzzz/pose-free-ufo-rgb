import torch

from inference import concatenate_chunk_targets


def test_concatenate_chunk_targets_preserves_order():
    chunks = []
    for chunk_index in range(4):
        inputs = {
            "context_image": torch.full((1, 1, 1), chunk_index),
            "target_camtoworlds": torch.full((1, 2, 1, 4, 4), chunk_index),
            "target_time": torch.tensor([[chunk_index * 2, chunk_index * 2 + 1]]),
            "height": 160,
        }
        targets = {
            "target_image": torch.full((1, 2, 1, 3, 2, 2), chunk_index),
            "target_frame_idx": torch.tensor(
                [[chunk_index * 2 + 1, chunk_index * 2 + 2]]
            ),
        }
        chunks.append((inputs, targets))

    render_input, targets = concatenate_chunk_targets(chunks)

    assert render_input["context_image"].item() == 3
    assert render_input["height"] == 160
    assert render_input["target_camtoworlds"].shape[1] == 8
    assert render_input["target_time"].tolist() == [list(range(8))]
    assert targets["target_image"].shape[1] == 8
    assert targets["target_frame_idx"].tolist() == [list(range(1, 9))]
