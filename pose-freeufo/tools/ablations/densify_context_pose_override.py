#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--input", type=Path, required=True)
p.add_argument("--start", type=int, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()


def project_R(M):
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def so3_log(R):
    c = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    th = np.arccos(c)
    if th < 1e-9:
        return np.zeros(3)
    return th / (2.0 * np.sin(th)) * np.array([
        R[2,1] - R[1,2],
        R[0,2] - R[2,0],
        R[1,0] - R[0,1],
    ])


def so3_exp(w):
    th = np.linalg.norm(w)
    if th < 1e-9:
        return np.eye(3)
    k = w / th
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0],
    ])
    return np.eye(3) + np.sin(th)*K + (1-np.cos(th))*(K@K)


def interp_pose(A, B, alpha):
    out = np.eye(4)
    dR = A[:3,:3].T @ B[:3,:3]
    w = so3_log(dR)
    out[:3,:3] = A[:3,:3] @ so3_exp(alpha * w)
    out[:3,3] = (1-alpha)*A[:3,3] + alpha*B[:3,3]
    return out


with np.load(a.input, allow_pickle=False) as x:
    scene_name = str(x["scene_name"].item())
    frame_ids = x["frame_ids"].astype(int)
    camera_ids = x["camera_ids"].astype(str)
    poses = x["omega_camera_to_world_rig_local"].astype(np.float64)
    Ks = x["predicted_intrinsics_ufo"].astype(np.float64)
    scale = float(x["rgb_metric_scale"]) if "rgb_metric_scale" in x else np.nan

cams = sorted(set(camera_ids), key=lambda z: {"1":0,"0":1,"2":2}.get(z, 99))
ref_cam = "0"

pose_map = {
    (int(f), str(c)): T
    for f, c, T in zip(frame_ids, camera_ids, poses)
}

ctx_frames = sorted(
    f for f in set(frame_ids)
    if (int(f), ref_cam) in pose_map
)

if len(ctx_frames) < 2:
    raise RuntimeError(f"need >=2 context frames, got {ctx_frames}")

print("context frames:", ctx_frames)

# 固定多相机 rig：由所有 context 预测结果估计。
rig = {}
for cam in cams:
    if cam == ref_cam:
        rig[cam] = np.eye(4)
        continue

    rels = []
    for f in ctx_frames:
        if (f, cam) in pose_map:
            rels.append(
                np.linalg.inv(pose_map[(f, ref_cam)]) @ pose_map[(f, cam)]
            )

    R = project_R(sum(T[:3,:3] for T in rels))
    t = np.median(
        np.stack([T[:3,3] for T in rels]), axis=0
    )

    X = np.eye(4)
    X[:3,:3] = R
    X[:3,3] = t
    rig[cam] = X

# 每个 camera 的 intrinsics 用 context 预测中值。
K_cam = {}
for cam in cams:
    arr = Ks[camera_ids == cam]
    K_cam[cam] = np.median(arr, axis=0)

dense_frames = list(range(a.start, a.start + 20))

dense_ref = {}
for f in dense_frames:
    if f in ctx_frames:
        dense_ref[f] = pose_map[(f, ref_cam)]
        continue

    if f < ctx_frames[-1]:
        for i in range(len(ctx_frames)-1):
            f0, f1 = ctx_frames[i], ctx_frames[i+1]
            if f0 <= f <= f1:
                alpha = (f-f0) / float(f1-f0)
                dense_ref[f] = interp_pose(
                    pose_map[(f0, ref_cam)],
                    pose_map[(f1, ref_cam)],
                    alpha,
                )
                break
    else:
        # 最后一段：constant SE(3) velocity extrapolation
        f0, f1 = ctx_frames[-2], ctx_frames[-1]
        alpha = (f-f0) / float(f1-f0)
        dense_ref[f] = interp_pose(
            pose_map[(f0, ref_cam)],
            pose_map[(f1, ref_cam)],
            alpha,
        )

out_f, out_c, out_T, out_K, out_role = [], [], [], [], []

for f in dense_frames:
    for cam in cams:
        out_f.append(f)
        out_c.append(cam)
        out_T.append(dense_ref[f] @ rig[cam])
        out_K.append(K_cam[cam])
        out_role.append(
            "context" if f in ctx_frames else "trajectory_predicted"
        )

a.output.parent.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    a.output,
    scene_name=np.asarray(scene_name),
    scope=np.asarray("all"),
    coordinate_frame=np.asarray("rig_local_metric"),
    metric_scale_source=np.asarray("moge2_gca_context_rgb_only"),
    world_gauge=np.asarray("first_context_front_camera"),
    target_rgb_used_for_camera=np.asarray(False),

    frame_ids=np.asarray(out_f, dtype=np.int32),
    camera_ids=np.asarray(out_c),
    roles=np.asarray(out_role),

    omega_camera_to_world_rig_local=np.asarray(out_T, dtype=np.float32),
    predicted_intrinsics_ufo=np.asarray(out_K, dtype=np.float32),

    rgb_metric_scale=np.asarray(scale),
)

print("================================")
print("CONTEXT-ONLY DENSE POSE PASS")
print("frames :", len(set(out_f)))
print("poses  :", len(out_f))
print("target RGB used:", False)
print("output :", a.output)
print("================================")
