# Scene621：VGGT-Omega Pose 替换 UFO 实验

这是一份面向组会汇报的完整说明。实验目标是回答：在同一个 UFO checkpoint、同一个场景和同一组 RGB 下，把 Waymo GT 相机外参换成 VGGT-Omega 预测外参，渲染质量会下降多少？

## 1. 一页结论

我们没有把 VGGT-Omega 和 UFO 联合训练。VGGT-Omega 只作为 UFO 前面的相机位姿估计器：

```text
RGB -> VGGT-Omega 预测 pose -> 坐标对齐 -> UFO 使用预测 pose 建场景/渲染
```

UFO 从零开始，只在 Waymo scene621 上训练 10,000 optimizer steps；训练过程全部使用 GT pose。VGGT-Omega 使用官方 1B/512 预训练权重，训练完成后才离线预测 pose。

| 实验 | Context pose | Target pose | PSNR | Delta PSNR | SSIM |
| --- | --- | --- | ---: | ---: | ---: |
| E0 | GT | GT | 24.190 dB | 0.000 dB | 0.8163 |
| E1 | Omega context-only | GT | 17.837 dB | -6.353 dB | 0.3639 |
| E2 | Omega all-frame | Omega all-frame | 22.433 dB | -1.757 dB | 0.7311 |

主要观察：E1 下降 6.35 dB，但 E2 只下降 1.76 dB。说明 UFO 不仅要求单张图的 pose 准确，还非常依赖 context pose 和 target pose 是否处于一致的相机轨迹中。

## 2. 两个模型分别做什么

### UFO

UFO 是最终的新视角渲染模型。它接收：

- Context RGB：用于理解和重建场景的参考图像；
- Context pose：每张参考图像从哪里、朝哪里拍摄；
- Target pose：希望从哪个相机位置渲染；
- 相机内参。

UFO 根据 context 建立 3D Gaussian 场景，然后从 target pose 渲染 RGB。

### VGGT-Omega

VGGT-Omega 在本实验中只根据多张 RGB 预测相机 pose。它不生成最终 RGB、不参与 UFO loss，也不向 UFO 反向传播梯度。

## 3. 模型是如何初始化的

### UFO 初始化

UFO 没有使用之前未训练完成的 checkpoint，而是随机初始化后只训练 scene621：

```text
train_scene_indices = 621
num_iterations = 10000
resume_from = null
auto_resume = false
```

训练日志确认：

```text
Loaded 1 annotations.
Training from scratch. No checkpoint found.
Starting training from iteration 0 to 10000
```

最终 checkpoint 是 `ckpt_009999.pth`，内部记录 `latest_step=9999`、scheduler step `10000`、`resume_from=None`。

### VGGT-Omega 初始化

VGGT-Omega 不在 scene621 上训练，直接加载官方 `vggt_omega_1b_512.pt`，以 BF16 inference 模式预测 pose。

### 每组 UFO 推理的场景初始化

E0/E1/E2 都加载同一个 UFO 10k checkpoint，并从空的 `scene = {}` 开始递推建立 Gaussian 场景。三组之间不共享场景状态。

## 4. Context、Target 和 All-frame

Context 是提供给 UFO 建场景的参考图像。Target 是 UFO 要渲染，并与 GT RGB 计算指标的图像。

scene621 的固定 2 秒窗口为：

```text
context frame 0  -> target frame 1, 2, 3, 4
context frame 5  -> target frame 6, 7, 8, 9
context frame 10 -> target frame 11, 12, 13, 14
context frame 15 -> target frame 16, 17, 18, 19
```

每个时间点使用 3 个相机：

- Context：4 个时间点 x 3 相机 = 12 张 RGB；
- Target：16 个时间点 x 3 相机 = 48 张 RGB；
- All-frame：12 张 context + 48 张 target = 60 张 RGB。

Context-only 指 Omega 只看 12 张 context RGB，只预测这 12 张图的 pose。All-frame 指 Omega 一次看完 60 张 RGB，并预测全部 60 个 pose。

## 5. 原始 UFO 与当前代码的区别

### 原始 UFO

原始 Dataset 直接从 Waymo annotation 读取 GT：

```python
scene_json["camera_to_world"][camera][frame_idx]
```

然后转换成 UFO 使用的 canonical 坐标：

```text
canonical_to_flu
  @ world_to_canonical
  @ camera_to_world
  @ opencv2dataset
```

原始代码没有替换指定 frame/camera pose 的接口。

### 当前改动一：导出 UFO 真正使用的帧

新增 `tools/export_ufo_pose_manifest.py`。它直接运行真实 `UFODataset`，导出 frame id、camera id、context/target 角色、RGB 路径、GT pose 和坐标约定。

这样不会人工猜测 UFO 选择了哪些帧，Omega 和 UFO 使用的是完全相同的图像集合。

### 当前改动二：Omega pose exporter

VGGT-Omega 仓库新增 `tools/export_ufo_pose_override.py`：

```text
manifest
  -> 读取 RGB
  -> VGGTOmega(images)
  -> encoding_to_camera(pose_enc)
  -> Omega raw w2c
  -> 求逆得到 raw c2w
  -> GT Sim(3) 对齐
  -> 转成 UFO/Waymo camera_to_world
  -> omega_pose_override.npz
```

输出同时保留 raw pose、aligned pose、GT pose、frame/camera id 和 Sim(3) 参数，便于检查。

### 当前改动三：直线轨迹 Sim(3) 修复

Omega 自己的世界坐标没有固定原点、方向和米制尺度，必须求统一尺度、旋转和平移：

```text
X_waymo = scale * R * X_omega + t
```

第一版只用相机中心做 Umeyama。但 scene621 接近直线行驶，只看一条直线无法确定绕直线的旋转，曾出现 ATE 很小、绝对旋转误差却约 72.8 度。

当前实现先用相机朝向确定全局旋转，再用相机中心确定尺度和平移。修复后的 context-only 平均旋转误差为 1.387 度，ATE RMSE 为 0.0727 m。

### 当前改动四：UFO Dataset pose override

新增 `PoseOverrideStore`，从下面的位置按 `(scene_name, frame_id, camera_id)` 精确读取 pose：

```text
<override_root>/<scene_name>/omega_pose_override.npz
```

`UFODataset` 增加三种模式：

```text
none:    context=GT,    target=GT
context: context=Omega, target=GT
all:     context=Omega, target=Omega
```

替换只发生在 Dataset 返回 `camera_to_world` 的位置。UFO Transformer、Gaussian head、`update_scene()`、renderer 和 loss 都没有改。

### 当前改动五：完整指标

旧 inference 虽然递推 4 个 chunk，但只计算最后一个 chunk 的指标。当前版本会在最终累计 scene 上渲染全部 16 个 target 时间点，也就是 48 张图，再统一计算指标。

## 6. 三组实验具体做了什么

### E0：GT 基线

```text
Context pose = GT
Target pose = GT
```

Omega 不参与。它表示 scene621 10k UFO 在准确相机参数下的表现。

### E1：Omega context-only

```text
Omega 输入 = 12 张 context RGB
Context pose = Omega
Target pose = GT
Target RGB = 只用于评分，不进入 Omega
```

E1 测量输入 pose 误差对 UFO 的直接影响。它是最重要、信息边界最干净的一组，但 context GT pose 仍参与 Sim(3) gauge 对齐，所以还不是完全无 GT 部署。

### E2：Omega all-frame

```text
Omega 输入 = 12 张 context RGB + 48 张 target RGB
Context pose = Omega
Target pose = Omega
```

E2 测量 context 和 target 全部换成同一套 Omega pose 后，系统一致性能恢复多少。由于 Omega 看过 target RGB，并且 target GT pose 参与 Sim(3) 对齐，E2 只能称为“全部替换外参诊断”，不能称为严格 pose-free NVS。

## 7. 完整指标

| 实验 | PSNR | SSIM | Occupied PSNR | Dynamic PSNR | Dynamic SSIM | Depth RMSE | Dynamic Depth RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E0 | 24.190 | 0.8163 | 24.167 | 18.820 | 0.5441 | 3.603 m | 5.470 m |
| E1 | 17.837 | 0.3639 | 17.883 | 16.527 | 0.2543 | 7.448 m | 8.080 m |
| E2 | 22.433 | 0.7311 | 22.402 | 18.570 | 0.4995 | 4.011 m | 5.625 m |

Omega pose 诊断：

| Scope | 图像数 | 平均旋转误差 | ATE RMSE | 平均 RPE 旋转 | 峰值显存 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Context-only | 12 | 1.387 deg | 0.0727 m | 0.0632 deg | 7.39 GiB |
| All-frame | 60 | 1.600 deg | 0.0783 m | 0.0493 deg | 10.40 GiB |

## 8. 结果解释

E1 中，UFO 根据 Omega context pose 把图像内容放入 3D 世界，却从 GT target pose 渲染。几厘米和约 1.4 度的误差足以造成建筑边缘、纹理和物体轮廓的像素错位。这个 UFO 又是在 scene621 上单场景拟合的，对 pose 很敏感，因此下降 6.35 dB。

E2 中，建场景和渲染都使用同一次 Omega 推理的 pose。整套坐标仍有误差，但 context 和 target 的误差更一致，因此恢复了 E1 损失中的大部分，只比 E0 低 1.76 dB。

当前最稳妥的结论是：

> VGGT-Omega pose 可以支撑 UFO 运行，但 UFO 不只要求单帧 pose 小误差，还要求 context 和 target 的相机轨迹保持一致。

## 9. 实验限制和下一步

- 目前只有一个训练场景和一个固定 2 秒窗口；
- UFO 在 scene621 上训练并在 scene621 上评估，是场景内拟合实验；
- E2 使用了 target RGB 和 target GT Sim(3)，不是严格未知目标视角；
- 当前只替换外参，仍使用 Waymo GT 内参；
- 正式结论需要多个窗口、多个场景的均值和方差。

建议下一步优先做：同一个 scene621 checkpoint 上评估多个 `start_idx`，确认 E1/E2 的下降是否稳定；随后扩展到多个 scene-specific checkpoint。

## 10. 组会展示文件

所有展示文件集中在：

```text
outputs/scene621_group_meeting/
```

其中三段视频均为纯 render，无 GT、无文字标注。每段为 16 帧、10 FPS、720 x 160；一帧横向排列 3 个 240 x 160 相机视图。

