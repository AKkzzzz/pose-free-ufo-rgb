import torch

from ufo.utils.losses import compute_depth_loss, compute_sky_depth_loss


def test_empty_depth_supervision_is_differentiable_zero():
    pred = torch.ones(1, 1, 2, 2, requires_grad=True)
    target = torch.zeros_like(pred)

    loss = compute_depth_loss(pred, target)

    assert loss.item() == 0.0
    loss.backward()
    assert pred.grad is not None
    assert torch.count_nonzero(pred.grad) == 0


def test_empty_sky_supervision_is_differentiable_zero():
    pred = torch.ones(1, 1, 1, 2, 2, requires_grad=True)
    sky_mask = torch.zeros_like(pred)

    depth_loss, flow_loss = compute_sky_depth_loss(pred, sky_mask)

    assert depth_loss.item() == 0.0
    assert flow_loss.item() == 0.0
    depth_loss.backward()
    assert pred.grad is not None
    assert torch.count_nonzero(pred.grad) == 0
