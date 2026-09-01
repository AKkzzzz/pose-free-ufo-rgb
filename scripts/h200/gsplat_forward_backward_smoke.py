#!/usr/bin/env python3
import json
import torch
from gsplat.rendering import rasterization


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device("cuda")
    means = torch.tensor([[-0.2, 0.0, 2.0], [0.2, 0.0, 2.2], [0.0, 0.2, 2.4]], device=device, requires_grad=True)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 3, device=device, requires_grad=True)
    scales = torch.full((3, 3), 0.08, device=device, requires_grad=True)
    opacities = torch.full((3,), 0.8, device=device, requires_grad=True)
    colors = torch.tensor([[1.0, 0.1, 0.1], [0.1, 1.0, 0.1], [0.1, 0.1, 1.0]], device=device, requires_grad=True)
    viewmats = torch.eye(4, device=device)[None]
    intrinsics = torch.tensor([[[80.0, 0.0, 32.0], [0.0, 80.0, 32.0], [0.0, 0.0, 1.0]]], device=device)
    rendered, alpha, _ = rasterization(
        means, quats, scales, opacities, colors, viewmats, intrinsics,
        width=64, height=64, packed=False,
    )
    loss = rendered.square().mean() + alpha.mean()
    loss.backward()
    gradients = {name: float(value.grad.norm()) for name, value in {
        "means": means, "quats": quats, "scales": scales,
        "opacities": opacities, "colors": colors,
    }.items()}
    if not all(torch.isfinite(value.grad).all() for value in (means, quats, scales, opacities, colors)):
        raise RuntimeError("Non-finite gsplat gradients")
    print(json.dumps({
        "device": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "render_shape": list(rendered.shape),
        "loss": float(loss.detach()),
        "gradient_norms": gradients,
        "GSPLAT_FORWARD_BACKWARD": "PASS",
    }, indent=2))


if __name__ == "__main__":
    main()
