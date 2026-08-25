# Scene621：VGGT-Omega 相机参数替换 UFO 实验

这是一份面向组会汇报的完整说明。实验目标是回答：在同一个 UFO checkpoint、同一个场景和同一组 RGB 下，把 Waymo GT 相机外参 T 和内参 K 换成 VGGT-Omega 预测值，渲染质量会下降多少？

## 1. 一页结论

我们没有把 VGGT-Omega 和 UFO 联合训练。VGGT-Omega 只作为 UFO 前面的相机位姿估计器：

```text
RGB -> VGGT-Omega 预测 T 和 K -> 坐标/分辨率适配 -> UFO 建场景/渲染
```

UFO 从零开始，只在 Waymo scene621 上训练 10,000 optimizer steps；训练过程全部使用 GT pose。VGGT-Omega 使用官方 1B/512 预训练权重，训练完成后才离线预测 pose。

| 实验 | 外参 T | 内参 K | 长序列 PSNR | Delta PSNR | SSIM |
| --- | --- | --- | ---: | ---: | ---: |
| E0 | GT | GT | 24.478 dB | 0.000 dB | 0.7805 |
| E1 | Omega | GT | 22.594 dB | -1.884 dB | 0.7014 |
| E2 | GT | Omega | 23.542 dB | -0.935 dB | 0.7438 |
| E3 | Omega | Omega | 23.504 dB | -0.973 dB | 0.7354 |

主要观察：只替换外参下降 1.88 dB，只替换内参下降 0.94 dB，说明 UFO 对外参更敏感；但完整替换只下降 0.97 dB，反而比只替换外参好 0.91 dB，说明同一次 Omega 推理产生的 T/K 一致性很重要。

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

P0/P1/P2 都加载同一个 UFO 10k checkpoint，并从空的 `scene = {}` 开始递推建立 Gaussian 场景。三组之间不共享场景状态。

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

## 6. 早期三组外参诊断

### P0：GT 基线

```text
Context pose = GT
Target pose = GT
```

Omega 不参与。它表示 scene621 10k UFO 在准确相机参数下的表现。

### P1：Omega context-only

```text
Omega 输入 = 12 张 context RGB
Context pose = Omega
Target pose = GT
Target RGB = 只用于评分，不进入 Omega
```

P1 测量输入 pose 误差对 UFO 的直接影响。它是信息边界最干净的一组，但 context GT pose 仍参与 Sim(3) gauge 对齐，所以还不是完全无 GT 部署。

### P2：Omega all-frame

```text
Omega 输入 = 12 张 context RGB + 48 张 target RGB
Context pose = Omega
Target pose = Omega
```

P2 测量 context 和 target 全部换成同一套 Omega pose 后，系统一致性能恢复多少。由于 Omega 看过 target RGB，并且 target GT pose 参与 Sim(3) 对齐，P2 只能称为“全部替换外参诊断”，不能称为严格 pose-free NVS。

## 7. 完整指标

| 实验 | PSNR | SSIM | Occupied PSNR | Dynamic PSNR | Dynamic SSIM | Depth RMSE | Dynamic Depth RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P0 | 24.190 | 0.8163 | 24.167 | 18.820 | 0.5441 | 3.603 m | 5.470 m |
| P1 | 17.837 | 0.3639 | 17.883 | 16.527 | 0.2543 | 7.448 m | 8.080 m |
| P2 | 22.433 | 0.7311 | 22.402 | 18.570 | 0.4995 | 4.011 m | 5.625 m |

Omega pose 诊断：

| Scope | 图像数 | 平均旋转误差 | ATE RMSE | 平均 RPE 旋转 | 峰值显存 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Context-only | 12 | 1.387 deg | 0.0727 m | 0.0632 deg | 7.39 GiB |
| All-frame | 60 | 1.600 deg | 0.0783 m | 0.0493 deg | 10.40 GiB |

## 8. 结果解释

P1 中，UFO 根据 Omega context pose 把图像内容放入 3D 世界，却从 GT target pose 渲染。几厘米和约 1.4 度的误差足以造成建筑边缘、纹理和物体轮廓的像素错位。这个 UFO 又是在 scene621 上单场景拟合的，对 pose 很敏感，因此下降 6.35 dB。

P2 中，建场景和渲染都使用同一次 Omega 推理的 pose。整套坐标仍有误差，但 context 和 target 的误差更一致，因此恢复了 P1 损失中的大部分，只比 P0 低 1.76 dB。

当前最稳妥的结论是：

> VGGT-Omega pose 可以支撑 UFO 运行，但 UFO 不只要求单帧 pose 小误差，还要求 context 和 target 的相机轨迹保持一致。

## 9. 实验限制和下一步

- 目前只有一个训练场景；短窗口和整段长序列结果都属于 scene621 场景内拟合；
- UFO 在 scene621 上训练并在 scene621 上评估，是场景内拟合实验；
- P2 以及最终矩阵的 E1/E3 使用了 target RGB 和 target GT Sim(3)，不是严格未知目标视角；
- 当前只替换外参，仍使用 Waymo GT 内参；
- 正式结论需要多个窗口、多个场景的均值和方差。

建议下一步优先扩展到多个 scene-specific checkpoint，报告跨场景均值和方差。

## 10. 组会展示文件

所有展示文件集中在：

```text
outputs/scene621_group_meeting/
```

原来的三段短视频均为纯 render，无 GT、无文字标注。每段为 16 帧、10 FPS、720 x 160；一帧横向排列 3 个 240 x 160 相机视图。

## 11. scene621 整段长序列

为了观察短窗口之外的稳定性，新增整段 198 帧结果。实现采用 10 个滑动窗口，起点为：

```text
0, 20, 40, 60, 80, 100, 120, 140, 160, 178
```

每个窗口使用该时间段内的 4 个 context frame 独立建立 UFO scene，渲染窗口内全部 20 个相机时刻；窗口之间清空 UFO scene state，最后按原始 frame id 去重并拼接。末窗口保留 180--197，因此最终严格覆盖 scene621 的 0--197 共 198 帧。

这里的“长序列”是完整场景的滑窗重建展示，不代表 UFO 在一个隐状态里连续记忆了 19.8 秒。这样做与训练时 4 chunk / 20 frame 的时间范围一致，也避免无限累积 Gaussian token 超出训练分布并占满 4090 显存。

三条视频仍然只有模型 render，横向顺序为 camera 1 / camera 0 / camera 2（左前 / 正前 / 右前），没有 GT、标签或说明文字。规格统一为 198 帧、19.8 秒、10 FPS、720 x 160。

| 长序列外参诊断 | PSNR | SSIM | Dynamic PSNR | 相对 P0 PSNR |
| --- | ---: | ---: | ---: | ---: |
| P0 GT context + GT render pose | 24.48 | 0.7805 | 19.83 | 0.00 |
| P1 Omega context + GT render pose | 18.10 | 0.3677 | 17.13 | -6.38 |
| P2 Omega context + Omega render pose | 22.59 | 0.7014 | 19.45 | -1.88 |

长序列还暴露出一个短视频看不到的现象：P0 在 frame 80--139 的窗口约为 26--27 dB，但 frame 160 以后约为 21 dB。说明 scene621 单场景训练 10k 后，不同时间段的拟合质量仍不均匀。

长序列展示目录：

```text
outputs/scene621_group_meeting/long_sequence/
├── long_sequence_metrics.csv
├── long_sequence_metrics.json
├── E0_GT/E0_GT_full_scene_render_3cam.mp4
├── E0_GT/metrics.json
├── E1_Omega_context/E1_Omega_context_full_scene_render_3cam.mp4
├── E1_Omega_context/metrics.json
├── E2_Omega_all/E2_Omega_all_full_scene_render_3cam.mp4
└── E2_Omega_all/metrics.json
```

每个 `metrics.json` 同时包含整段汇总和 10 个窗口的分段指标；每个 `frame_mapping.json` 记录视频帧到原始 scene frame 的逐帧映射。

## 12. 最终四组 T/K 标定矩阵

最终实验把外参 T 和内参 K 作为两个独立变量。四组都使用同一个 scene621 10k checkpoint、同样的 198 帧滑窗协议；Omega 组使用 all-frame 输出。

Omega 在 416 x 624 图像上解码 K，而 UFO 使用 160 x 240。适配代码按每张图真实执行的中心裁剪、resize 和 padding，把 K 先逆变换回磁盘图像 320 x 480，再映射到 UFO 160 x 240。scene621 没有实际裁剪或 padding，最终主点从 Omega 的 `(312, 208)` 变为 UFO 的 `(120, 80)`。

全序列 Omega K 相对 Waymo GT 的平均误差为：焦距 7.21%，FoV 2.60 度，主点 2.81 px。Omega 主点固定在图像中心，并不代表它预测了任意主点。

| 实验 | 外参 T | 内参 K | PSNR | Delta PSNR | SSIM | Dynamic PSNR |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| E0 | GT | GT | 24.478 | 0.000 | 0.7805 | 19.834 |
| E1 | Omega | GT | 22.594 | -1.884 | 0.7014 | 19.449 |
| E2 | GT | Omega | 23.542 | -0.935 | 0.7438 | 19.604 |
| E3 | Omega | Omega | 23.504 | -0.973 | 0.7354 | 19.624 |

核心结论：外参误差单独影响更大；但 T/K 误差不是简单相加。E3 比 E1 高 0.91 dB，表明 Omega 自己输出的外参与内参配套使用时，比 Omega 外参混用 GT 内参更一致。当前可以说“GT 相机标定可被基本替换”，但必须同时说明仍使用 GT Sim(3) 对齐，且 all-frame Omega 看过 target RGB。

最终四组文件统一放在：

```text
outputs/scene621_group_meeting/long_sequence/camera_matrix/
├── camera_matrix_metrics.csv
├── camera_matrix_metrics.json
├── E0_E1_E2_E3_camera_matrix_render_only.mp4
├── E0_GT_T_GT_K/E0_GT_T_GT_K_render_3cam.mp4
├── E1_Omega_T_GT_K/E1_Omega_T_GT_K_render_3cam.mp4
├── E2_GT_T_Omega_K/E2_GT_T_Omega_K_render_3cam.mp4
└── E3_Omega_T_Omega_K/E3_Omega_T_Omega_K_render_3cam.mp4
```

## 13. 动态物体不动的 D0 / D1 定位实验

长视频中车辆在一个 20 帧窗口内近似静止、到下一个窗口突然跳到新位置。为区分长视频 target pose 复用错误和 object assignment collapse，固定同一个 scene621 10k checkpoint，增加两个只在 inference 生效的诊断模式：

```text
D0 predicted：使用 checkpoint 原本预测的 bbox assignment
D1 stable-track bbox Oracle：最终 Gaussian 中心落在 GT 3D bbox 内时，硬分配给该物体；否则分配为背景
```

D1 不改网络、不训练，也不使用 LiDAR。它仍使用原版 `context_instances_pose -> target_instances_pose` 搬运 Gaussian，只把 assignment 换成 GT bbox 几何 Oracle。长序列实现会先取完整 20 帧都有效的物体槽位，并在最终 scene token 更新和 Gaussian 重解码之后重新判框，避免缺失槽位的 identity pose 和陈旧 assignment 污染结果。由于中途进入、离开、遮挡或 annotation 暂缺的轨迹都被当作背景，D1 是 stable-track Oracle，测得的收益是完整 Oracle 收益的下界。

| Window | 模式 | Target pose 相邻帧平移 | 背景概率 | 硬动态比例 | Gaussian 平均位移 | PSNR | Dynamic PSNR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0--19 | D0 predicted | 0.236 m | 0.999985 | 0.000% | 0.000029 m | 24.125 | 18.743 |
| 0--19 | D1 stable-track Oracle | 0.236 m | 0.996495 | 0.350% | 0.005532 m | 24.272 | 18.932 |
| 140--159 | D0 predicted | 0.153 m | 0.999544 | 0.000% | 0.000860 m | 23.449 | 18.820 |
| 140--159 | D1 stable-track Oracle | 0.153 m | 0.988231 | 1.177% | 0.006919 m | 23.833 | 19.420 |

判断属于情况 B，而不是长视频 target pose repeat：两个窗口的 GT object pose 都逐帧变化，但 D0 硬 assignment 全为背景，最终 Gaussian 基本不动。D1 在 start=140 上使 PSNR 提升 0.384 dB、Dynamic PSNR 提升 0.601 dB，并产生合理的逐帧物体位移，因此 object assignment 是当前动态链路的第一故障点。

`a7bface` 初版文档曾把 collapse 归因于 local `gs_token_means` 和 global bbox 错配。该判断混淆了 stage2 的局部中间量和 `forward_renderer()` 的最终 class-loss 输入，现已撤回。`update_scene()` 会先把 token means 转到 global scene，render 前再用累计 scene 重新 stage2；class loss 实际使用的是 global `gs_token_means` 和 global `context_instances_corner`。不能把训练 GT 直接改成 `context_instances_corner_local`。

为了在 class-loss 真正发生的位置验证，新增 renderer-time 坐标诊断。它保持 recurrent `update_scene(render=True)` 链路，对当前 chunk 的同一批 token 分别计算 global/local bbox 标签，并同时检查最终 Gaussian 是否落框。

| Window | Chunk | 当前 token 的 global 正样本 | local 正样本 | Gaussian global 正样本 | Gaussian 正样本比例 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0--19 | 0 / 1 / 2 / 3 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 788 / 859 / 833 / 803 | 0.684%--0.746% |
| 140--159 | 0 / 1 / 2 / 3 | 1 / 1 / 0 / 0 | 0 / 1 / 0 / 0 | 2025 / 2083 / 2035 / 1945 | 1.688%--1.808% |

start=140 的正式累计 class loss 中只有 1--2 个动态 token，且 checkpoint 对这些正样本的 foreground recall 为 0%。与此同时，同一时刻有约 2,000 个 Gaussian 落在车辆框内。因此现有证据更支持：64 个 Gaussian 聚合成一个 8×8 scene-token 均值后，车辆正样本消失或变得极端稀疏；随后严重类别不平衡和 coarse ownership 使 assignment head collapse。它仍是诊断结论，不等于已经证明唯一根因；scene update 后 ownership 是否陈旧仍需单独实验。

现在可以确定和不能确定的边界是：

```text
已确定：target object pose 逐帧变化
已确定：checkpoint assignment 近乎全部 background
已确定：stable-track bbox Oracle 改善动态指标
已确定：renderer class-loss 使用 global token + global bbox
已确定：Gaussian-level 有正样本，但 token-level 正样本为 0 或极少

尚未确定：token-level supervision 改造后是否足以解决全部动态问题
尚未确定：scene update 后旧 token 的 ownership 是否需要重新分配
```

本轮没有修改训练 GT、没有启动重训。

Oracle 几何自检通过：start=0 被分配 Gaussian 到 bbox 中心的最大距离为 5.37 m，所涉 bbox 最大半对角线为 5.61 m；object pose 旋转正交误差最大为 `1.8e-7`。这排除了早期调试中 identity slot 导致的百米假位移。

诊断文件位于：

```text
outputs/scene621_group_meeting/dynamic_assignment_diagnostic/
├── dynamic_assignment_summary.csv
├── renderer_coordinates/
│   ├── renderer_assignment_coordinates_summary.csv
│   ├── renderer_assignment_coordinates_start_000.json
│   └── renderer_assignment_coordinates_start_140.json
├── start_000/
│   ├── dynamic_assignment_comparison.json
│   ├── D_predicted_start_000_render_3cam.mp4
│   ├── D_oracle_bbox_start_000_render_3cam.mp4
│   └── D0_predicted_top_D1_oracle_bottom_start_000_render_3cam.mp4
└── start_140/
    ├── dynamic_assignment_comparison.json
    ├── D_predicted_start_140_render_3cam.mp4
    ├── D_oracle_bbox_start_140_render_3cam.mp4
    └── D0_predicted_top_D1_oracle_bottom_start_140_render_3cam.mp4
```

对照视频仍是纯 render：上半部分为 D0，下半部分为 D1；每一行横向排列 camera 1 / camera 0 / camera 2，没有 GT 或文字覆盖。
