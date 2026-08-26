import copy

import torch

from ufo.models.vit import Transformer
from ufo.models.archs.small import (
    expand_spatial_token_assignments,
    points_in_boxes_probability,
    transform_gaussian_means_with_instances,
    transform_gaussian_quats_with_instances,
)
from ufo.utils.losses import compute_lifespan_reg_loss
from ufo.utils.misc import detach_tensors


def test_paper_lifespan_regularizer_is_inverse_beta():
    beta = torch.tensor([1.0, 2.0, 4.0]).reshape(1, 1, 1, 1, 3, 1)
    loss = compute_lifespan_reg_loss({"lifespan": beta}, "paper_beta")
    torch.testing.assert_close(loss, torch.tensor((1.0 + 0.5 + 0.25) / 3.0))


def test_detach_tensors_preserves_values_and_breaks_history():
    source = torch.tensor([2.0], requires_grad=True)
    scene = {"state": source.square(), "nested": [source + 1]}
    detached = detach_tensors(scene)
    assert detached["state"].grad_fn is None
    assert detached["nested"][0].grad_fn is None
    torch.testing.assert_close(detached["state"], scene["state"])


def test_sequential_backward_matches_summed_detached_chunks():
    torch.manual_seed(7)
    baseline = torch.nn.Linear(4, 3)
    sequential = copy.deepcopy(baseline)
    chunks = [(torch.randn(2, 4), torch.randn(2, 3)) for _ in range(4)]

    sum(torch.nn.functional.mse_loss(baseline(x), y) for x, y in chunks).backward()
    for x, y in chunks:
        torch.nn.functional.mse_loss(sequential(x), y).backward()

    for expected, actual in zip(baseline.parameters(), sequential.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad)


def test_transformer_checkpoint_path_has_finite_gradients():
    transformer = Transformer(embed_dim=16, depth=2, num_heads=4, grad_checkpointing=True)
    transformer.train()
    x = torch.randn(2, 5, 16, requires_grad=True)
    transformer(x).square().mean().backward()
    assert torch.isfinite(x.grad).all()


def test_bbox_yaw_rotates_gaussian_quaternion():
    context = torch.eye(4).reshape(1, 1, 1, 4, 4)
    target = context.clone()
    target[0, 0, 0, :2, :2] = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    quat = torch.tensor([1.0, 0.0, 0.0, 0.0]).reshape(1, 1, 1, 4)
    probability = torch.ones(1, 1, 1, 1)
    actual = transform_gaussian_quats_with_instances(context, target, quat, probability)
    expected = torch.tensor([2**-0.5, 0.0, 0.0, 2**-0.5]).reshape(1, 1, 1, 1, 4)
    torch.testing.assert_close(actual, expected)


def test_stable_bbox_delta_keeps_identity_transform_exact():
    poses = torch.eye(4).reshape(1, 1, 1, 4, 4).repeat(1, 1, 33, 1, 1)
    means = torch.tensor([300.123, -200.456, 50.789]).reshape(1, 1, 1, 3)
    probabilities = torch.softmax(torch.randn(1, 1, 1, 33), dim=-1)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = transform_gaussian_means_with_instances(
            poses, poses, means, probabilities, stable_delta=True
        )
    torch.testing.assert_close(actual, means[:, None], rtol=0, atol=0)


def test_points_in_boxes_reports_dynamic_positive_and_masks_padding():
    corners = torch.tensor([
        [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0], [1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0],
        [-1.0, 1.0, 1.0], [1.0, 1.0, 1.0],
    ])
    boxes = torch.stack([corners, corners + 10.0]).unsqueeze(0)
    points = torch.tensor([[[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]]])
    probabilities = points_in_boxes_probability(
        points, boxes, torch.tensor([[1, 0]]), temperature=0.01
    )
    assert probabilities[0, 0].argmax() == 1
    assert probabilities[0, 1].argmax() == 0
    assert probabilities[..., 2].eq(0).all()


def test_token_assignment_broadcast_matches_spatial_patch():
    token_weights = torch.arange(6).reshape(1, 1, 6, 1).float()
    actual = expand_spatial_token_assignments(
        token_weights, views=1, height=4, width=6, patch_size=2
    )[0, 0, 0, ..., 0]
    expected = torch.tensor([
        [0, 0, 1, 1, 2, 2],
        [0, 0, 1, 1, 2, 2],
        [3, 3, 4, 4, 5, 5],
        [3, 3, 4, 4, 5, 5],
    ]).float()
    torch.testing.assert_close(actual, expected)

