# 1. 总体流程：VGGT-Omega 如何接到 UFO 前面

## 一句话说明

VGGT-Omega 是相机位姿前端，UFO 是场景重建和新视角渲染模型。两个模型没有联合训练：

```text
RGB 图像
  -> VGGT-Omega 预测相机外参
  -> 对齐到 Waymo 世界坐标系
  -> UFO Dataset 用预测外参替换 GT 外参
  -> UFO 建立 Gaussian 场景并渲染目标图像
```

## 两个模型各自做什么

### UFO

UFO 接收参考图像及其相机参数，根据参考图像建立 3D Gaussian 场景，再从目标相机位置渲染 RGB。它需要知道：

- 图像内容是什么；
- 每张参考图像从哪里、朝哪个方向拍摄；
- 最终要从哪个相机位置渲染。

### VGGT-Omega

VGGT-Omega 在本实验中只根据多张 RGB 预测每张图像的相机外参。它不生成最终 RGB，也不参与 UFO 的 loss 或反向传播。

## 三种不同的“初始化”

### UFO 权重初始化

UFO 从随机权重开始，只在 scene621 上训练 10,000 optimizer steps。训练没有加载旧 checkpoint：

```text
resume_from = null
auto_resume = false
```

最终模型是 `ckpt_009999.pth`。

### VGGT-Omega 权重初始化

VGGT-Omega 不重新训练，直接加载官方 `vggt_omega_1b_512.pt`，以 BF16 inference 方式预测 pose。

### UFO 推理时的场景初始化

每组 E0/E1/E2 都加载相同的 UFO 10k checkpoint，并从空的 `scene = {}` 开始递推建立 Gaussian 场景。不会读取此前实验的场景状态。

## Context 和 Target

Context 是 UFO 已知的参考图像，Target 是 UFO 要渲染并与 GT RGB 比较的图像。

scene621 的固定 2 秒窗口为：

```text
context frame 0  -> target frame 1, 2, 3, 4
context frame 5  -> target frame 6, 7, 8, 9
context frame 10 -> target frame 11, 12, 13, 14
context frame 15 -> target frame 16, 17, 18, 19
```

每个时间点使用 3 个相机，因此：

- Context：4 个时间点 x 3 相机 = 12 张 RGB；
- Target：16 个时间点 x 3 相机 = 48 张 RGB；
- All-frame：12 + 48 = 60 张 RGB。

## 本实验替换了什么

只替换相机外参 pose。没有替换：

- 相机内参；
- RGB；
- UFO 网络或 renderer；
- Gaussian head；
- 训练 loss；
- Waymo 物体框。

Omega 预测的内参会保存在输出文件中，但 UFO 仍使用 Waymo GT 内参，以便只测外参误差。

