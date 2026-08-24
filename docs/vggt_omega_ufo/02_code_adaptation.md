# 2. 代码适配：原始 UFO 与当前版本的差异

## 原始数据链路

原始 UFO Dataset 直接从 Waymo annotation 读取：

```python
scene_json["camera_to_world"][camera][frame_idx]
```

然后转换为 UFO 使用的 canonical 坐标：

```text
canonical_to_flu
  @ world_to_canonical
  @ camera_to_world
  @ opencv2dataset
```

原始代码没有接口可以为指定 frame/camera 替换 pose。

## 改动 1：导出 UFO 实际使用的图像

新增 `tools/export_ufo_pose_manifest.py`。该脚本运行真正的 `UFODataset`，导出：

- scene name；
- frame id；
- camera id；
- context/target 角色；
- RGB 绝对路径；
- GT `camera_to_world`；
- OpenCV 坐标约定下的 GT `c2w`。

这样 Omega 使用的图像集合与 UFO 完全一致，不需要手工猜帧号。

## 改动 2：Omega pose exporter

VGGT-Omega 仓库新增 `tools/export_ufo_pose_override.py`：

```text
manifest
  -> 读取指定 RGB
  -> VGGTOmega(images)
  -> encoding_to_camera(pose_enc)
  -> raw w2c
  -> inverse 得到 raw c2w
  -> GT Sim(3) 对齐
  -> 转回 UFO/Waymo camera_to_world
  -> omega_pose_override.npz
```

输出同时保留 raw、aligned 和 GT pose，便于后续诊断。

## 改动 3：修复直线轨迹的 Sim(3) 退化

最初只用相机中心做 Umeyama。scene621 近似直线行驶，只看一条直线无法确定绕直线的全局旋转，曾出现：

```text
ATE 很小，但绝对旋转误差约 72.8 度
```

当前实现改为：

1. 用相机朝向拟合全局旋转；
2. 固定旋转后，用相机中心拟合尺度和平移。

修复后的 context-only 指标为平均旋转误差 `1.387` 度、ATE RMSE `0.0727 m`。

## 改动 4：UFO pose override 存储层

新增 `ufo/dataset/pose_override.py`。它读取：

```text
<override_root>/<scene_name>/omega_pose_override.npz
```

并按 `(frame_id, camera_id)` 建立 pose 映射。缺场景、缺相机、缺 frame 或重复 key 都会直接报错，避免静默用错 pose。

## 改动 5：Dataset 按角色选择 pose

`UFODataset` 新增：

```text
pose_override_mode = none | context | all
pose_override_dir
```

选择规则：

```text
none:
  context -> GT
  target  -> GT

context:
  context -> Omega
  target  -> GT

all:
  context -> Omega
  target  -> Omega
```

替换发生在 Dataset 返回 `camera_to_world` 的位置，后续 canonicalization、Plucker ray、Gaussian 建场景和 renderer 都沿用原始 UFO 流程。

## 改动 6：Inference 参数和 checkpoint 契约

`inference.py` 增加了：

- `--annotation_file`；
- `--pose_override_dir`；
- `--pose_override_mode`。

推理会先读取 JSON config，再从 checkpoint 保存的训练 Namespace 补齐 JSON 中没有显式写出的默认参数。这样模型初始化与训练时一致，不会漏掉 `paper_affine_transform` 或 `object_assignment_gt_mode`。

## 改动 7：完整四段指标

旧 inference 虽然递推 4 个 chunk，但只对最后一个 chunk 的 target 计算指标。当前版本会合并四段 target pose 和 GT，在最终累计 Gaussian scene 上渲染全部 16 个 target 时间点，即 48 张图。

## 没有改动的部分

- `small.py` 的核心模型结构；
- `update_scene()` 的递推逻辑；
- renderer；
- 训练 loss；
- Omega 权重；
- UFO 与 Omega 之间没有梯度传递。

