# VGGT-Omega 1B 论文六项 benchmark 复现报告

## 结论

本机已经对论文列出的六项 benchmark 全部完成一次真实的 VGGT-Omega 1B 前向推理与指标计算：

- 静态：7Scenes、NRGBD、ETH3D
- 动态：DyCheck、Sintel、TUM-Dynamic

但是这不是可以声称为“严格重现作者表格”的最终版本。论文只公开了“每个 scene/sequence 随机取 10 帧”和指标定义，官方仓库没有发布评估代码、随机 seed、完整 scene split、depth mask 细节或尺度对齐实现。因此当前结果是固定 seed、可重复、尽量贴近论文描述的近似复现。

Waymo 不属于这六项 benchmark。Waymo 结果独立标记为论文同类深度指标的自定义 LiDAR 诊断，不参与论文表格汇总，也不称为官方 Waymo benchmark。

## 当前结果

使用 checkpoint：`checkpoints/vggt_omega_1b_512.pt`

SHA256：`c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934`

| Dataset | AUC@3 当前 / 论文 | AUC@30 当前 / 论文 | delta1.25 当前 / 论文 | AbsRel 当前 / 论文 |
|---|---:|---:|---:|---:|
| 7Scenes | 32.6 / 29.6 | 84.3 / 83.1 | 96.1 / 94.6 | 0.049 / 0.058 |
| NRGBD | 71.0 / 89.7 | 96.1 / 98.8 | 100.0 / 99.6 | 0.007 / 0.010 |
| ETH3D | 42.2 / 49.8 | 91.4 / 88.5 | 99.9 / 99.8 | 0.017 / 0.012 |
| DyCheck | 47.4 / 38.4 | 89.8 / 87.3 | 96.7 / 98.4 | 0.047 / 0.038 |
| Sintel | 32.9 / 35.3 | 69.8 / 73.0 | 87.2 / 89.5 | 0.249 / 0.097 |
| TUM-Dynamic | 44.5 / 30.2 | 86.4 / 82.3 | 97.6 / 97.4 | 0.035 / 0.041 |

所有表中数值是 scene/sequence 指标的算术平均。AUC 和 delta1.25 以百分数表示。

## 协议

- 模型：官方 `VGGTOmega`，官方 `load_and_preprocess_images(..., image_resolution=512)`，官方 `encoding_to_camera()`。
- 输入：每个 scene/sequence 固定随机抽取 10 帧，seed 为 0；同一场景的 10 帧一次前向。
- pose：预测和 GT 都转换为 world-to-camera；对所有帧对计算相对旋转和 translation direction angular error，取两者较大值；用整数角度 histogram 计算 AUC@3 和 AUC@30。
- depth：仅在有效 GT depth 像素计算；每个 10 帧 clip 使用单个 median depth ratio 尺度；计算 AbsRel 与 delta1.25。
- 预处理：GT depth 复现官方图像的极端宽高比中心裁剪，并 nearest-neighbor resize 到模型 depth 输出大小。
- 显存：RTX 4090 48 GB，10 帧一次前向峰值 reserved memory 约 7.75 GiB。

## 数据来源

- Sintel：官方 MPI Sintel training images、depth 和 camera data。
- TUM-Dynamic：官方 Freiburg3 sitting/walking 的 8 个动态序列。
- DyCheck：官方 iPhone 数据的 7 个标准场景：apple、block、paper-windmill、space-out、spin、teddy、wheel。每场景直接从官方 Drive 下载固定 seed 对应的 10 个 RGB/depth/camera 三元组。
- 7Scenes、NRGBD、ETH3D：使用 `HarrisonPENG/SpatialBenchmark` 的 medium 统一格式作为数据适配层，再从每个场景固定抽 10 帧。它不是 VGGT-Omega 作者公布的 split。

## 主要偏差

- 论文未公开随机 seed。10 帧抽样对 pose AUC 尤其敏感，TUM 和 Sintel 的不同运动区间差异很大。
- 论文未公开 depth 尺度对齐和有效 mask 的实现。当前使用每 clip 单尺度，不使用 per-frame oracle scale。
- SpatialBenchmark 的 7Scenes、NRGBD、ETH3D medium 子集不是作者的隐藏抽样列表。
- Sintel `temple_3` 在当前抽样下 AbsRel 为 3.477，是总体 AbsRel 偏高的主要来源；`ambush_2` 也明显异常。没有事后删序列或挑 seed。
- TUM 曾用另一套相对位姿组合方向做过审计，得到 AUC@3/AUC@30 = 0.0/2.1，明显与论文协议不符。该失败结果保留为 `tum_dynamic_seed0_official_auc.json`，不进入最终表。

## Waymo 自定义诊断

Waymo 已按论文公开的“每 scene 随机抽 10 帧，并只将这 10 帧联合输入模型”的设置重跑。旧的 `AbsRel 0.709 / delta1.25 25.5%` 来自先推理完整 198 帧再抽输出，并且像素对齐和尺度公式也与当前六项复现不一致，因此不再作为有效结果。

- Waymo validation scenes 068/100/172：pose scene mean AUC@3 `92.10`、AUC@30 `97.28`；depth scene mean AbsRel `0.985`、delta1.25 `37.64%`。
- 用户指定的 training scenes 552/172/621：depth scene mean AbsRel `0.067`、delta1.25 `94.61%`，但 Waymo 是论文训练来源，不能将 training 数字作为无偏泛化 benchmark。

完整协议、逐场景结果和命令见 `WAYMO_PAPER_STYLE_EVALUATION_CN.md`。Waymo 仍是论文同指标的自定义诊断，不属于论文六项官方 benchmark。

## 最终结果文件

目录：`../results/vggt_omega/paper_benchmarks/`

- `7scenes_seed0.json`
- `nrgbd_seed0.json`
- `eth3d_seed0.json`
- `dycheck_seed0.json`
- `sintel_seed0_final.json`
- `tum_dynamic_seed0_final.json`

不带 `_final` 的 Sintel/TUM 文件以及 `*_official_auc.json` 是协议调试审计记录，不用于最终表。

## 运行命令

```bash
CUDA_VISIBLE_DEVICES=0 python tools/evaluate_spatialbench_paper_protocol.py \
  --root medium \
  --checkpoint checkpoints/vggt_omega_1b_512.pt \
  --output_dir ../results/vggt_omega/paper_benchmarks \
  --datasets 7scenes nrgbd eth3d --seed 0

python tools/download_dycheck_eval.py \
  --output benchmarks/dycheck \
  --scenes apple block paper-windmill space-out spin teddy wheel --seed 0

CUDA_VISIBLE_DEVICES=0 python tools/evaluate_dycheck_paper_protocol.py \
  --root benchmarks/dycheck \
  --checkpoint checkpoints/vggt_omega_1b_512.pt \
  --output ../results/vggt_omega/paper_benchmarks/dycheck_seed0.json

CUDA_VISIBLE_DEVICES=0 python tools/evaluate_sintel_paper_protocol.py \
  --root benchmarks/sintel \
  --checkpoint checkpoints/vggt_omega_1b_512.pt \
  --output ../results/vggt_omega/paper_benchmarks/sintel_seed0_final.json

CUDA_VISIBLE_DEVICES=0 python tools/evaluate_tum_dynamic_paper_protocol.py \
  --root benchmarks/tum_dynamic \
  --checkpoint checkpoints/vggt_omega_1b_512.pt \
  --output ../results/vggt_omega/paper_benchmarks/tum_dynamic_seed0_final.json
```
