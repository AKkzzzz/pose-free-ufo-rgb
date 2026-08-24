# VGGT-Omega 到 UFO 文档索引

本目录记录 scene621 单场景训练和 VGGT-Omega pose 替换实验。建议按顺序阅读：

1. [01_pipeline_for_beginners.md](01_pipeline_for_beginners.md)：面向初学者的总体流程和术语说明。
2. [02_code_adaptation.md](02_code_adaptation.md)：原始代码与当前代码的差异、数据格式和坐标转换。
3. [03_experiment_design_and_interpretation.md](03_experiment_design_and_interpretation.md)：E0/E1/E2 的设计、结果和正确解读。
4. [04_reproduction_guide.md](04_reproduction_guide.md)：模型、输出位置和完整复现命令。

数值结果的简表也保存在上一级目录的
[scene621_vggt_omega_pose_results.md](../scene621_vggt_omega_pose_results.md)。

## 当前版本

- UFO 分支：`exp/vggt-omega-pose`
- UFO 结果文档 commit：`c74ef84`
- VGGT-Omega 分支：`exp/ufo-pose-export`
- VGGT-Omega exporter commit：`5fd36c3`
- 数据：Waymo training index `621`
- UFO checkpoint：`ckpt_009999.pth`

这里的 checkpoint 和实验图片位于本地 ignored 的 `outputs/`，不在 GitHub 中。

