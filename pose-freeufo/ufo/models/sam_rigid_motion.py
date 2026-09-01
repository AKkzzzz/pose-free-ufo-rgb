import torch


@torch.no_grad()
def fit_track_rigid_velocity(
    means,
    track_ids,
    gs_time,
    timespan,
    min_pixels=32,
    max_speed=30.0,
):
    """
    means:     [B,T,V,H,W,3] global-frame Gaussian centers
    track_ids: [B,T,V,H,W], 0 = background
    gs_time:   [B,T,V] normalized time
    timespan:  seconds

    Returns:
        rigid_velocity: same shape as means
        diagnostics: dict
    """
    assert means.ndim == 6 and means.shape[-1] == 3
    assert track_ids.shape == means.shape[:-1], (
        f"SAM/Gaussian mismatch: tracks={track_ids.shape}, means={means.shape}"
    )
    assert gs_time.shape == means.shape[:3], (
        f"time mismatch: time={gs_time.shape}, means={means.shape}"
    )

    B, T, V, H, W, _ = means.shape
    velocity = torch.zeros_like(means)

    total_tracks = 0
    fitted_tracks = 0
    speeds = []

    for b in range(B):
        ids = torch.unique(track_ids[b])
        ids = ids[ids > 0]

        total_tracks += int(ids.numel())

        for track_id in ids:
            centers = []
            times = []

            # Treat every camera/time observation independently.
            for t in range(T):
                for v in range(V):
                    mask = track_ids[b, t, v] == track_id
                    n = int(mask.sum())

                    if n < min_pixels:
                        continue

                    pts = means[b, t, v][mask]

                    # Robust object center against boundary/background leakage.
                    center = pts.median(dim=0).values

                    centers.append(center)
                    times.append(gs_time[b, t, v] * float(timespan))

            if len(centers) < 2:
                continue

            centers = torch.stack(centers)       # [N,3]
            times = torch.stack(times).float()   # [N]

            # Need at least two genuinely different timestamps.
            if (times.max() - times.min()).abs() < 1e-5:
                continue

            t_mean = times.mean()
            c_mean = centers.mean(dim=0)

            dt = times - t_mean
            denom = (dt * dt).sum()

            if denom < 1e-8:
                continue

            # Least-squares translation velocity.
            v_obj = (
                dt[:, None] * (centers - c_mean)
            ).sum(dim=0) / denom

            speed = torch.linalg.vector_norm(v_obj)

            # Reject pathological pseudo-track geometry.
            if not torch.isfinite(speed):
                continue

            if speed > max_speed:
                v_obj = v_obj * (max_speed / speed.clamp_min(1e-6))
                speed = torch.tensor(
                    max_speed,
                    device=speed.device,
                    dtype=speed.dtype,
                )

            # Every Gaussian owned by this track shares ONE translation velocity.
            mask_all = track_ids[b] == track_id
            velocity[b][mask_all] = v_obj

            fitted_tracks += 1
            speeds.append(speed)

    if speeds:
        speeds = torch.stack(speeds)
        speed_mean = float(speeds.mean())
        speed_max = float(speeds.max())
    else:
        speed_mean = 0.0
        speed_max = 0.0

    diagnostics = {
        "sam_track_count": total_tracks,
        "sam_fitted_track_count": fitted_tracks,
        "sam_rigid_speed_mean": speed_mean,
        "sam_rigid_speed_max": speed_max,
        "sam_dynamic_ratio": float((track_ids > 0).float().mean()),
    }

    return velocity, diagnostics


def apply_sam_temporal_gate(
    opacity,
    track_ids,
    delta_t,
    sigma=0.25,
):
    """
    Gaussian temporal gate for dynamic tracks only.

    Static background track_id=0 remains unchanged.

    opacity:   [...]
    track_ids: [...]
    delta_t:   [...] seconds
    """
    assert track_ids.shape == delta_t.shape

    dynamic = track_ids > 0

    gate = torch.ones_like(delta_t, dtype=opacity.dtype)

    if dynamic.any():
        sigma = max(float(sigma), 1e-4)
        gate_dynamic = torch.exp(
            -0.5 * (delta_t[dynamic] / sigma) ** 2
        )
        gate[dynamic] = gate_dynamic

    while gate.ndim < opacity.ndim:
        gate = gate.unsqueeze(-1)

    return opacity * gate
