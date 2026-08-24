# 3. E0/E1/E2 实验设计与结果解读

## E0：GT pose 基线

```text
Context pose = GT
Target pose  = GT
```

Omega 不参与。这表示 scene621 10k UFO checkpoint 在准确相机参数下的正常表现。

## E1：Context-only Omega

Omega 只读取 12 张 context RGB，只预测这 12 张图的 pose：

```text
Context RGB  -> Omega pose -> UFO 建场景
Target pose  -> GT pose    -> UFO 渲染
Target RGB   -> 仅用于评分
```

E1 回答：参考图像 pose 出现 Omega 级别的误差，而目标渲染位置仍准确时，UFO 会损失多少质量？

E1 不把 target RGB 交给 Omega，因此是最重要的 pose sensitivity 实验。但 context GT pose 仍参与 Sim(3) gauge 对齐，所以它不是完全不使用 GT 的部署方案。

## E2：All-frame Omega

Omega 一次读取全部 60 张 context + target RGB，并预测全部 pose：

```text
Context pose = Omega all-frame pose
Target pose  = Omega all-frame pose
```

E2 回答：当建场景和渲染都使用同一套 Omega pose 时，系统的一致性能恢复多少？

E2 不是严格 pose-free NVS，因为 Omega 看过 target RGB，而且 target GT pose 参与了 Sim(3) 对齐。它只能作为“全部替换外参”的诊断。

## 实验结果

| 实验 | Context pose | Target pose | PSNR | 相对 E0 | SSIM | Depth RMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| E0 | GT | GT | 24.190 dB | 0.000 dB | 0.8163 | 3.603 m |
| E1 | Omega context-only | GT | 17.837 dB | -6.353 dB | 0.3639 | 7.448 m |
| E2 | Omega all-frame | Omega all-frame | 22.433 dB | -1.757 dB | 0.7311 | 4.011 m |

## 如何理解

E1 中，UFO 根据 Omega context pose 把参考图像内容放入 3D 世界，但最后从 GT target pose 渲染。几厘米和约 1.4 度的误差会造成纹理、建筑边缘和物体轮廓的像素错位。scene621 单场景过拟合模型尤其敏感，因此 PSNR 下降 6.35 dB。

E2 中，context 和 target 都使用同一次 Omega 推理得到的 pose。整套坐标可能仍有误差，但建场景和渲染相机的误差更一致，因此恢复了 E1 损失中的大部分，只比 E0 低 1.76 dB。

当前最稳妥的结论是：

> VGGT-Omega pose 能支撑 UFO 运行，但 UFO 不只要求单帧 pose 误差小，还非常依赖 context 与 target 相机轨迹的一致性。

## 不能过度解读的地方

- 目前只有一个训练场景和一个固定 2 秒窗口；
- UFO 在同一个 scene621 上训练和评估，属于场景内拟合实验；
- E2 使用 target RGB 和 target GT Sim(3)，不能当作未知目标视角性能；
- 这次只替换外参，没有测 Omega 内参；
- 要得到正式结论，还需要多个窗口和多个场景统计均值及方差。

