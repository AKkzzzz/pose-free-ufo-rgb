# VGGT-Omega Waymo 论文同指标诊断

## 结论

VGGT-Omega 论文没有报告 Waymo 测试指标，也没有发布 Waymo 专用评测协议或目标数字。Waymo 在论文中是训练数据来源之一。因此本文结果只能称为“按照论文公开的 10 帧输入设置和同类指标，在本地 Waymo GT 上进行的自定义诊断”，不能称为官方 Waymo benchmark。

本次修正了旧评测的关键设置错误：旧运行将完整 198 帧一次送入模型，再从输出中抽 10 帧计算指标；本次先从场景抽取 10 帧，只把这 10 帧联合送入 VGGT-Omega。模型使用全局注意力，所以这两种输入不是等价的。

## 评测协议

- 模型：官方 `VGGTOmega` 和 `vggt_omega_1b_512.pt`。
- 输入：Waymo 单个 scene 的 `FRONT` 相机，即 camera 0。
- 抽样：每个 scene 随机抽 10 帧，scene seed 为 `seed + scene_index`。
- 前向：同一 scene 的 10 张 RGB 一次联合前向，不输入 Waymo pose、内参、外参或 LiDAR。
- 图像预处理：官方 `load_and_preprocess_images(..., image_resolution=512)`，实际 tensor 为 `10 x 3 x 416 x 624`。
- pose：用官方 `encoding_to_camera()` 得到 world-to-camera；与 Waymo ego pose 和相机外参组成的 GT world-to-camera 比较所有 45 个帧对，报告 AUC@3 和 AUC@30。
- depth：使用 `depth_flows_4` 中投影到相机的稀疏 LiDAR z-depth；全 10 帧共用一个 `median(gt / pred)` 尺度；报告 AbsRel 和 delta1.25。
- GT 对齐：LiDAR depth 按官方 RGB 预处理的裁剪规则处理，并 nearest-neighbor resize 到模型 depth 输出大小。
- 置信度：主指标不使用 `depth_conf` 过滤，避免人为挑选容易像素。

## Waymo Validation

这是更适合观察泛化的主结果。

| Scene | 采样帧数 | AUC@3 | AUC@30 | AbsRel | delta1.25 |
|---|---:|---:|---:|---:|---:|
| 068 | 10 | 91.85 | 98.81 | 0.825 | 20.88% |
| 100 | 10 | 100.00 | 100.00 | 0.908 | 52.90% |
| 172 | 10 | 84.44 | 93.04 | 1.222 | 39.14% |
| scene mean | 10 | 92.10 | 97.28 | 0.985 | 37.64% |

结果文件：`../results/vggt_omega/waymo_paper_style_validation_seed0/`

validation 上 pose 很强，但 metric depth 与稀疏 LiDAR 相差明显。将输入改成连续帧 50-59 后，pose scene mean 进一步达到 AUC@3 `96.05`、AUC@30 `99.60`，但 depth 仍为 AbsRel `1.159`、delta1.25 `40.02%`。因此 validation depth 失败不能简单归因于随机 10 帧间隔过大。

连续 10 帧结果：`../results/vggt_omega/waymo_contiguous10_validation/`

## 指定的 Training Scenes

按用户指定运行了 552、172、621。Waymo 是论文训练数据来源，因此这些数字可能受训练域或具体训练样本重叠影响，不应作为无偏泛化结果。

| Scene | AUC@3 | AUC@30 | AbsRel | delta1.25 |
|---|---:|---:|---:|---:|
| 552 | 92.59 | 97.19 | 0.068 | 93.38% |
| 172 | 20.74 | 80.22 | 0.066 | 96.49% |
| 621 | 18.52 | 36.74 | 0.068 | 93.96% |
| scene mean | 43.95 | 71.38 | 0.067 | 94.61% |

结果文件：`../results/vggt_omega/waymo_paper_style_seed0_final/`

training 和 validation 的 depth 差距非常大，而 validation pose 反而更稳定。当前最合理的解释是：training 结果不适合作为泛化 benchmark；同时仍不能排除本地两个 split 的预处理分布存在差异。已人工检查 LiDAR 投影覆盖，点位与对应 RGB 物体和道路基本对齐，没有发现明显的帧号错配。

## 旧 Scene 100 结果为何错误

旧结果 `AbsRel 0.709 / delta1.25 25.5%` 来自 `official_demo_full_video/waymo-100`，存在以下协议问题：

1. 实际输入模型的是完整 198 帧，不是论文描述的 10 帧输入。
2. 旧代码把预测 depth resize 回原始 GT，而不是把 GT 按官方 RGB crop/resize 映射到预测网格。
3. 旧尺度写成 `median(gt) / median(pred)`，当前六项论文复现统一使用 `median(gt / pred)`。

对 training scene 100 用相同帧号、正确 10 帧输入和当前评测代码重跑后得到：AUC@3 `31.11`、AUC@30 `88.52`、AbsRel `0.083`、delta1.25 `93.96%`。这确认旧的差数字主要是设置和评测实现造成的。

结果文件：`../results/vggt_omega/waymo_paper_style_scene100_seed0/`

## 运行命令

```bash
CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/vggt-omega/bin/python \
  tools/evaluate_waymo_paper_protocol.py \
  --data-root ../data/waymo_full \
  --split validation \
  --scenes 068 100 172 \
  --checkpoint checkpoints/vggt_omega_1b_512.pt \
  --output-dir ../results/vggt_omega/waymo_paper_style_validation_seed0 \
  --camera 0 --num-frames 10 --seed 0 --image-resolution 512
```

指定 training scenes：

```bash
CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/vggt-omega/bin/python \
  tools/evaluate_waymo_paper_protocol.py \
  --data-root ../data/waymo_full \
  --split training \
  --scenes 552 172 621 \
  --checkpoint checkpoints/vggt_omega_1b_512.pt \
  --output-dir ../results/vggt_omega/waymo_paper_style_seed0_final \
  --camera 0 --num-frames 10 --seed 0 --image-resolution 512
```

每个 scene 保存 `metrics.json` 和 `predictions.npz`；汇总保存在 `waymo_paper_style_metrics.json`。10 帧前向峰值 reserved GPU memory 为约 `7.71 GiB`。
