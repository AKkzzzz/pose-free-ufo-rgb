# VGGT-Omega 本地复现记录

## 当前环境

- GPU: NVIDIA GeForce RTX 4090，49140 MiB 显存
- 驱动/CUDA runtime: 550.163.01 / CUDA 12.4
- Conda 环境: `vggt-omega`
- Python/PyTorch: 3.10.19 / 2.4.1+cu121
- 模型参数量: 1,143,814,185
- 推理精度: RTX 4090 支持 bfloat16，模型会自动使用 BF16 autocast

`opencv-python` 在该无桌面容器中会因缺少 `libGL.so.1` 导入失败，因此环境使用兼容的
`opencv-python-headless==4.11.0.86`。

## 权重

权重是 Hugging Face 门控资源。先在
<https://huggingface.co/facebook/VGGT-Omega> 登录、同意条款并等待访问获批，然后执行：

```bash
conda activate vggt-omega
hf auth login
hf download facebook/VGGT-Omega vggt_omega_1b_512.pt \
  --local-dir checkpoints
```

不能用未获批账号下载。`VGGT-Omega-1B-512` 是重建实验需要的权重；256 版本是文本对齐模型。

本次由于服务器访问 Hugging Face 异常，512 权重实际从 ModelScope 的
`facebook/VGGT-Omega` 镜像下载。文件校验结果为：

```text
size:   4576706117 bytes
sha256: c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934
```

## 官方视频推理

```bash
conda activate vggt-omega
cd /inspire/hdd/project/intelligent-driving-agent/guoluosong-253108120129/workspace/yx/vggt-omega
python run_video_inference.py \
  --checkpoint checkpoints/vggt_omega_1b_512.pt \
  --video examples/forest_road.mp4 \
  --output-dir outputs/forest_road \
  --sample-fps 1 \
  --max-frames 100
```

输出包括：

- `scene.glb`: 带相机轨迹的彩色点云，可在浏览器/Blender 中查看
- `rgb_depth.mp4`: 输入 RGB 与预测逆深度并排视频
- `predictions.npz`: 深度、置信度、相机内外参
- `run_metadata.json`: 运行耗时、吞吐和峰值显存

`forest_road.mp4` 在 RTX 4090 上按 1 FPS 抽取 31 帧的实测结果：输入张量为
`31x3x384x688`，首次 HDD 权重加载 72.63 秒，模型前向 1.88 秒（16.45 FPS），峰值显存
8.28 GiB。输出均为有限值，GLB 包含约 100 万点和 31 个预测相机。

## Waymo 定性推理

Waymo 在论文中是训练数据来源之一，不是表 1/2 的评测集。论文未公布 Waymo 专用的数据加载、
采样比例或预处理代码。本地实验使用 validation scene 068 的 1920x1280 前视相机原图，每 10 帧
取一张，共 20 张，覆盖约 19 秒：

```bash
python run_video_inference.py \
  --checkpoint checkpoints/vggt_omega_1b_512.pt \
  --image-glob '/inspire/hdd/project/intelligent-driving-agent/guoluosong-253108120129/workspace/yx/data/waymo_full/datasets/waymo/validation/068/images/*_0.jpg' \
  --image-stride 10 \
  --max-frames 20 \
  --output-dir outputs/waymo_val_068_front
```

20 帧前向耗时 1.09 秒（18.40 FPS），峰值显存 7.42 GiB。所有输出均为有限值，GLB 含约
100 万点和 20 个相机。作为非论文诊断，将预测相机中心用单个 Sim(3) 对齐到 Waymo 标定真值后，
344.88 m 轨迹上的 ATE RMSE 为 0.96 m（轨迹长度的 0.28%）。

本地稀疏 LiDAR 深度与 VGGT 输出之间没有作者公布的 Waymo 评测协议。使用最近邻缩放和逐帧中值
尺度对齐得到的数值已保存到 `outputs/waymo_val_068_front/waymo_custom_metrics.json`，但不能与论文
表 2 比较，也不应称为官方 Waymo 指标。

官方 Gradio demo 可用下列命令启动：

```bash
python demo_gradio.py \
  --checkpoint checkpoints/vggt_omega_1b_512.pt \
  --image-resolution 512
```

## 论文指标与协议

论文表 1/2 使用每个场景或序列随机采样 10 帧，测试三个静态数据集 7 Scenes、NRGBD、ETH3D，
以及三个动态数据集 DyCheck、Sintel、TUM-Dynamic。当前公开仓库只包含推理和 demo，不包含论文
benchmark 的数据加载、尺度对齐、相机 AUC 或深度评测实现。因此，仅运行官方示例视频不能证明复现了论文指标。

公开的 1B 权重对应论文 `Ours-1B`，目标值如下：

| 数据集 | AUC@3 deg | AUC@30 deg | delta1.25 | AbsRel |
| --- | ---: | ---: | ---: | ---: |
| 7 Scenes | 29.6 | 83.1 | 94.6 | 0.058 |
| NRGBD | 89.7 | 98.8 | 99.6 | 0.010 |
| ETH3D | 49.8 | 88.5 | 99.8 | 0.012 |
| DyCheck | 38.4 | 87.3 | 98.4 | 0.038 |
| Sintel | 35.3 | 73.0 | 89.5 | 0.097 |
| TUM-Dynamic | 30.2 | 82.3 | 97.4 | 0.041 |

论文中更高的 Sintel `AUC@3=40.0`、`AUC@30=79.1`、`delta1.25=93.5`、`AbsRel=0.081`
属于未公开的 10B 模型，不能用公开 1B 权重复现。

完整定量复现还需要获得六个测试集、确定作者实际使用的随机 10 帧列表，并补齐与论文一致的相机
规范化和逐帧深度尺度对齐。否则随机采样和对齐细节会使结果与论文表格不可直接比较。
