# 4. Scene621 实验复现指南

## 仓库和分支

UFO：

```bash
git clone https://github.com/AKkzzzz/ufo-reproduction.git
cd ufo-reproduction
git checkout exp/vggt-omega-pose
```

VGGT-Omega：

```bash
git clone https://github.com/AKkzzzz/yxvggtfx.git vggt-omega
cd vggt-omega
git checkout exp/ufo-pose-export
```

两个仓库需要放在同一个父目录下，或通过 `OMEGA_ROOT` 指定 VGGT-Omega 仓库位置。

## 本地模型位置

UFO 10k checkpoint：

```text
outputs/scene621_10k/ufo_scene621_from_scratch_10k/checkpoints/ckpt_009999.pth
```

SHA256：

```text
68f2fbb169f657d73361ab5305fde58c37e1a8253f666fc60d64513a929d9344
```

VGGT-Omega checkpoint：

```text
../vggt-omega/checkpoints/vggt_omega_1b_512.pt
```

checkpoint 和输出没有提交到 GitHub，需要在执行机器上准备。

## 从零训练 scene621

在 UFO 仓库执行：

```bash
/root/miniconda3/envs/dggt_data/bin/python main.py \
  --config configs/experiments/ufo_scene621_10k_4090.json
```

关键日志应包含：

```text
Loaded 1 annotations.
Training from scratch. No checkpoint found.
Starting training from iteration 0 to 10000
```

本次 RTX 4090 实测：

- 总训练时间：3:48:46；
- 训练峰值显存：约 28.26 GiB；
- 每 1000 step 保存一次，共 10 个 checkpoint。

## 一键运行 E0/E1/E2

确认两个 checkpoint 都存在后，在 UFO 仓库执行：

```bash
bash scripts/vggt_omega/run_scene621_pose_experiment.sh
```

脚本会依次执行：

```text
1. 用 UFODataset 生成 scene621 manifest
2. Omega context-only 导出 12 个 pose
3. Omega all-frame 导出 60 个 pose
4. UFO E0
5. UFO E1
6. UFO E2
7. 汇总 summary.json
```

Omega 和 UFO 是不同进程，顺序运行，不会同时占用 GPU。

## 输出位置

```text
outputs/vggt_omega_pose/scene621_10k/
  scene621_manifest.json
  omega_context/<scene_name>/omega_pose_override.npz
  omega_context/<scene_name>/pose_metrics.json
  omega_all/<scene_name>/omega_pose_override.npz
  omega_all/<scene_name>/pose_metrics.json
  e0_gt/
  e1_omega_context/
  e2_omega_all/
  summary.json
```

每组 UFO 输出包含：

- 完整 16 帧、3 相机渲染视频；
- 48 张独立 PNG；
- PSNR、SSIM 和 depth RMSE JSON。

## 单独重跑某一组

E0 使用 `--pose_override_mode none`。

E1 使用：

```text
--pose_override_mode context
--pose_override_dir <omega_context>
```

E2 使用：

```text
--pose_override_mode all
--pose_override_dir <omega_all>
```

三组必须使用相同 checkpoint、scene、`start_idx`、内参、分辨率和 `filter_num`，否则 PSNR 差值不能归因于 pose。

