import torch


@torch.no_grad()
def fit_track_rigid_velocity(
    means,
    track_ids,
    gs_time,
    timespan,
    min_pixels=32,
    max_speed=30.0,
    min_observations=3,
):
    """
    Fit one shared translation velocity for each SAM track.

    Important:
    - background ID 0 is static
    - require >=3 temporal observations
    - pathological high-speed fits are REJECTED, not clamped
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
    rejected_fast = 0
    rejected_short = 0

    accepted_speeds = []
    raw_track_info = []

    for b in range(B):
        ids = torch.unique(track_ids[b])
        ids = ids[ids > 0]

        total_tracks += int(ids.numel())

        for track_id in ids:
            centers = []
            times = []

            for t in range(T):
                for v in range(V):
                    mask = track_ids[b, t, v] == track_id
                    n = int(mask.sum())

                    if n < min_pixels:
                        continue

                    pts = means[b, t, v][mask]

                    # Robust visible 3D center.
                    center = pts.float().median(dim=0).values

                    centers.append(center)
                    times.append(
                        (gs_time[b, t, v].float() * float(timespan))
                    )

            # Two-frame fitting is far too unstable for pseudo tracks.
            if len(centers) < min_observations:
                rejected_short += 1
                continue

            centers = torch.stack(centers)
            times = torch.stack(times)

            duration = float((times.max() - times.min()).abs())

            if duration < 1e-4:
                rejected_short += 1
                continue

            t_mean = times.mean()
            c_mean = centers.mean(dim=0)

            dt = times - t_mean
            denom = (dt * dt).sum()

            if denom < 1e-8:
                continue

            v_obj = (
                dt[:, None] * (centers - c_mean)
            ).sum(dim=0) / denom

            raw_speed = torch.linalg.vector_norm(v_obj)

            if not torch.isfinite(raw_speed):
                continue

            raw_speed_f = float(raw_speed)

            raw_track_info.append(
                (
                    raw_speed_f,
                    int(track_id),
                    len(centers),
                    duration,
                )
            )

            # CRITICAL:
            # Never turn a pathological 100 m/s fit into a valid 30 m/s motion.
            # Reject the track completely instead.
            if raw_speed_f > float(max_speed):
                rejected_fast += 1
                continue

            mask_all = track_ids[b] == track_id
            velocity[b][mask_all] = v_obj.to(velocity.dtype)

            fitted_tracks += 1
            accepted_speeds.append(raw_speed)

    if accepted_speeds:
        accepted = torch.stack(accepted_speeds).float()

        speed_mean = float(accepted.mean())
        speed_max = float(accepted.max())
        speed_p50 = float(torch.quantile(accepted, 0.50))
        speed_p90 = float(torch.quantile(accepted, 0.90))
    else:
        speed_mean = 0.0
        speed_max = 0.0
        speed_p50 = 0.0
        speed_p90 = 0.0

    # Print fastest raw fits so we can see whether 3D trajectory fitting is broken.
    raw_track_info.sort(reverse=True)

    top_text = ", ".join(
        f"id={tid}: {speed:.1f}m/s ({obs}obs,{duration:.2f}s)"
        for speed, tid, obs, duration in raw_track_info[:8]
    )

    print(
        f"[R4 rigid] tracks={total_tracks} "
        f"fitted={fitted_tracks} "
        f"reject_fast={rejected_fast} "
        f"reject_short={rejected_short} "
        f"speed_p50={speed_p50:.2f} "
        f"speed_p90={speed_p90:.2f} "
        f"speed_max={speed_max:.2f}"
    )

    if top_text:
        print(f"[R4 rigid] fastest raw: {top_text}")

    diagnostics = {
        "sam_track_count": total_tracks,
        "sam_fitted_track_count": fitted_tracks,
        "sam_rejected_fast_count": rejected_fast,
        "sam_rejected_short_count": rejected_short,
        "sam_rigid_speed_mean": speed_mean,
        "sam_rigid_speed_p50": speed_p50,
        "sam_rigid_speed_p90": speed_p90,
        "sam_rigid_speed_max": speed_max,
        "sam_dynamic_ratio": float(
            (track_ids > 0).float().mean()
        ),
    }

    return velocity, diagnostics



@torch.no_grad()
def rigidify_learned_velocity(
    forward_flow,
    track_ids,
    min_pixels=32,
    seed_speed=0.5,
    min_seed_count=4,
    min_seed_ratio=0.02,
    min_coherence=0.70,
    max_speed=30.0,
):
    """
    R4 motion propagation:

        sparse R3 Gaussian velocities
              ↓
        SAM object grouping
              ↓
        coherent motion seeds
              ↓
        one shared object velocity
              ↓
        propagate to all Gaussians of the track

    R3 velocity is intentionally sparse, therefore the full-mask
    velocity median must NOT be used.
    """

    assert forward_flow.shape[:-1] == track_ids.shape

    rigid_velocity = torch.zeros_like(forward_flow)

    total_tracks = 0
    moving_tracks = 0
    static_tracks = 0
    rejected_tracks = 0

    results = []

    B = forward_flow.shape[0]

    for b in range(B):

        ids = torch.unique(track_ids[b])
        ids = ids[ids > 0]

        total_tracks += int(ids.numel())

        for track_id in ids:

            mask = track_ids[b] == track_id
            n_total = int(mask.sum())

            if n_total < min_pixels:
                continue

            flow = forward_flow[b][mask].float()

            finite = torch.isfinite(flow).all(dim=-1)
            flow = flow[finite]

            if flow.shape[0] < min_pixels:
                continue

            speed = torch.linalg.vector_norm(flow, dim=-1)

            # ------------------------------------------------------
            # R3 is sparse:
            # only use meaningful velocity Gaussians as motion seeds.
            # ------------------------------------------------------
            seed_mask = speed > float(seed_speed)

            n_seed = int(seed_mask.sum())
            seed_ratio = n_seed / max(int(speed.numel()), 1)

            if n_seed < min_seed_count or seed_ratio < min_seed_ratio:
                static_tracks += 1
                continue

            seeds = flow[seed_mask]
            seed_speeds = torch.linalg.vector_norm(seeds, dim=-1)

            # ------------------------------------------------------
            # Estimate consensus motion direction.
            # ------------------------------------------------------
            directions = torch.nn.functional.normalize(
                seeds,
                dim=-1,
                eps=1e-6,
            )

            mean_direction = directions.mean(dim=0)
            coherence = torch.linalg.vector_norm(mean_direction)

            coherence_f = float(coherence)

            if coherence_f < min_coherence:
                rejected_tracks += 1
                continue

            consensus_direction = (
                mean_direction / coherence.clamp_min(1e-6)
            )

            # ------------------------------------------------------
            # Estimate robust scalar speed along consensus direction.
            #
            # This is more stable than component-wise median:
            # first determine direction, then robustly estimate speed.
            # ------------------------------------------------------
            projected_speed = (
                seeds * consensus_direction[None]
            ).sum(dim=-1)

            # Reject seeds pointing opposite to consensus.
            projected_speed = projected_speed[
                projected_speed > 0
            ]

            if projected_speed.numel() < min_seed_count:
                rejected_tracks += 1
                continue

            object_speed = projected_speed.median()

            object_speed_f = float(object_speed)

            if (
                not torch.isfinite(object_speed)
                or object_speed_f > max_speed
            ):
                rejected_tracks += 1
                continue

            v_obj = (
                consensus_direction * object_speed
            )

            # ------------------------------------------------------
            # Propagate ONE velocity to the entire SAM object.
            # ------------------------------------------------------
            rigid_velocity[b][mask] = v_obj.to(
                rigid_velocity.dtype
            )

            moving_tracks += 1

            results.append(
                (
                    object_speed_f,
                    int(track_id),
                    n_total,
                    n_seed,
                    seed_ratio,
                    coherence_f,
                )
            )

    if results:
        speeds = torch.tensor(
            [x[0] for x in results],
            dtype=torch.float32,
        )

        p50 = float(torch.quantile(speeds, 0.50))
        p90 = float(torch.quantile(speeds, 0.90))
        vmax = float(speeds.max())
    else:
        p50 = p90 = vmax = 0.0

    print(
        f"[R4 seed-rigid] "
        f"tracks={total_tracks} "
        f"moving={moving_tracks} "
        f"static={static_tracks} "
        f"rejected={rejected_tracks} "
        f"speed_p50={p50:.2f} "
        f"speed_p90={p90:.2f} "
        f"speed_max={vmax:.2f}"
    )

    # Most confident moving tracks.
    results.sort(key=lambda x: x[4], reverse=True)

    for (
        obj_speed,
        tid,
        n_total,
        n_seed,
        ratio,
        coherence,
    ) in results[:12]:

        print(
            f"[R4 seed-track] "
            f"id={tid} "
            f"N={n_total} "
            f"seeds={n_seed} "
            f"ratio={ratio:.3f} "
            f"speed={obj_speed:.3f} "
            f"coh={coherence:.3f}"
        )

    diagnostics = {
        "sam_track_count": total_tracks,
        "sam_moving_track_count": moving_tracks,
        "sam_static_track_count": static_tracks,
        "sam_rejected_track_count": rejected_tracks,
        "sam_rigid_speed_p50": p50,
        "sam_rigid_speed_p90": p90,
        "sam_rigid_speed_max": vmax,
    }

    return rigid_velocity, diagnostics


def apply_sam_temporal_gate(
    opacity,
    track_ids,
    delta_t,
    sigma=0.25,
):
    """
    Temporal gate for SAM dynamic tracks only.
    Background track_id=0 remains unchanged.
    """
    assert track_ids.shape == delta_t.shape

    dynamic = track_ids > 0
    sigma = max(float(sigma), 1e-4)

    # Compute stably in FP32, then cast back to renderer dtype.
    gate_dynamic = torch.exp(
        -0.5 * (delta_t.float() / sigma) ** 2
    ).to(dtype=opacity.dtype)

    gate = torch.ones_like(
        delta_t,
        dtype=opacity.dtype,
        device=opacity.device,
    )

    gate = torch.where(dynamic, gate_dynamic, gate)

    while gate.ndim < opacity.ndim:
        gate = gate.unsqueeze(-1)

    return opacity * gate

