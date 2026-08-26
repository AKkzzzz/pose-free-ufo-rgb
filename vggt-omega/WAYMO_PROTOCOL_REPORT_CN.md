# VGGT-Omega（本地 Waymo 前视协议）

当前结果由选中的 20 张 Waymo 原始前视 RGB 图像在一次联合前向中生成。它不是多个局部重建窗口的时间拼接，也不会把前一帧或前一个窗口的预测结果重新送回模型。

需要先明确：VGGT-Omega 官方仓库没有发布 Waymo 推理配置、数据转换脚本或 Waymo 指标评测协议。论文只说明 Waymo Open Dataset 是训练数据来源之一；论文中的公开定量表也没有把 Waymo 作为测试基准。因此，下面是我们为验证官方 checkpoint 而定义的本地协议，不应称为“官方 Waymo 协议”或直接与论文表格中的数值比较。

当前本地推理入口和复现记录在：

- `run_video_inference.py`
- `REPRODUCTION.md`
- 输出目录：`outputs/waymo_val_068_front/`

此前 `waymo_sparse_context_reconstruction/scene-068` 中的 MP4 使用了额外的
Sim(3) 对齐和点云重投影代码。由于 scene-068 的相机中心几何退化，全局 roll 未被约束，
视频出现斜向。该结果已在 `INVALID_ALIGNMENT.md` 中标记为无效，不能用于判断官方模型质量。

本次新增的官方 demo 全帧实验位于：
`../results/vggt_omega/official_demo_full_video/waymo-100/`。

## 当前配置

本报告顶部的 `outputs/waymo_val_068_front` 是早期 20 帧诊断；下面的官方 demo 结果是当前推荐的结果。

### 官方 demo 全帧 Waymo scene-100

- 官方入口：仓库 `demo_gradio.py` 的 `handle_uploads`、`run_model`、`predictions_to_glb`
- 相机：Waymo `*_0.jpg` 前视相机，1 个相机流
- 原始序列：198 帧，10 FPS，时间跨度 19.8 秒
- 官方采样：`sample-fps=10`，因此 198 帧全部作为输入
- 网络张量：`[1, 198, 3, 416, 624]`
- 前向次数：1 次，约 112.5 秒（RTX 4090）
- 输出：100 万点彩色 GLB、预测深度视频；没有额外 RGB 渲染器

文件：

- `01_official_sampled_context.mp4`：官方 demo 实际读取的 RGB 序列
- `02_official_reconstruction.glb`：官方深度反投影点云和预测相机
- `03_official_input_rgb_predicted_depth.mp4`：左侧输入 RGB，右侧预测 inverse-depth

这不是“少量 context 输入、渲染所有时间”的实验。VGGT-Omega 官方发布代码只为输入图像预测相机、深度和置信度，
不输出 3D Gaussian，也没有时间插值、新视角 RGB rasterization 或窗口间拼接。要实现 ReconDrive 风格的
部分帧输入/所有帧渲染，必须另外实现动态场景表示、位姿对齐和 rasterizer，不能把它称作官方 VGGT 推理。

- 数据集：Waymo Open Dataset validation scene `068`
- 场景属性：白天、Phoenix、晴天
- 相机：前视相机 1 个，即 camera ID `0`
- 原始分辨率：`1280 x 1920`（高 x 宽）
- 网络输入分辨率：`416 x 624`（高 x 宽）
- 原始帧率：10 FPS
- 抽帧间隔：每 10 帧取 1 帧，即送入模型的序列为 1 FPS
- 原始帧编号：`0, 10, 20, ..., 190`
- 输入数量：20 个时刻 x 1 个相机 = 20 张原始 RGB
- 覆盖时间：从 0 秒到 19 秒，首尾时间跨度为 19 秒
- 前向次数：1 次；20 张图像在同一次模型调用中联合处理
- 输入张量：加 batch 维前为 `[20, 3, 416, 624]`，模型中为 `[1, 20, 3, 416, 624]`
- checkpoint：官方 `vggt_omega_1b_512.pt`
- 精度：BF16 autocast
- 模型前向输入：只有 RGB 图像
- 不输入：Waymo 相机内参、相机外参、ego pose、LiDAR、3D 标注框、车辆速度、动态掩码或相机 ID

精确运行命令为：

```bash
python run_video_inference.py \
  --checkpoint checkpoints/vggt_omega_1b_512.pt \
  --image-glob '/inspire/hdd/project/intelligent-driving-agent/guoluosong-253108120129/workspace/yx/data/waymo_full/datasets/waymo/validation/068/images/*_0.jpg' \
  --image-stride 10 \
  --max-frames 20 \
  --output-dir outputs/waymo_val_068_front
```

对于这个 20 帧输入，具体关系是：

| 模型中的位置 | Waymo 原始帧 | 原始时间 | 模型为该输入图像预测的内容 |
|---:|---:|---:|---|
| 0 | 0 | 0 秒 | 深度、置信度、相机参数 |
| 1 | 10 | 1 秒 | 深度、置信度、相机参数 |
| 2 | 20 | 2 秒 | 深度、置信度、相机参数 |
| ... | ... | ... | ... |
| 18 | 180 | 18 秒 | 深度、置信度、相机参数 |
| 19 | 190 | 19 秒 | 深度、置信度、相机参数 |

原始 frame 1-9、11-19 等未抽中的图像既没有送入模型，也没有作为这次推理的监督目标。模型不会为这些中间时刻生成 RGB、深度或相机参数。

## 一次前向内部发生的过程

```text
Waymo 原始前视 RGB：frame 0、10、20、...、190
    ↓ balanced resize，保持宽高比并使 token 面积接近 512 x 512
20 张 624 x 416 RGB 堆叠为同一个序列
    ↓ 一次联合前馈
每张图像先进行帧内 attention，序列再进行跨帧 global/register attention
    ↓
相机头为每张输入图像预测 9 维 camera encoding
    ├─ 平移 3 维
    ├─ 四元数 4 维
    └─ 垂直/水平 FoV 2 维
稠密头为每张输入图像预测 depth 和 depth confidence
    ↓
将预测相机参数解码为 world-to-camera 外参和针孔相机内参
    ↓
用预测深度和预测相机反投影点云
    ↓
直接从对应输入 RGB 取点颜色，导出 scene.glb
    ↓
将输入 RGB 与彩色 inverse-depth 并排编码为 rgb_depth.mp4
```

所以这里的“视频效果”需要准确理解：

- `rgb_depth.mp4` 左侧是送入网络的原始 RGB，不是模型重建出来的新 RGB。
- 右侧是模型为同一输入图像预测的深度可视化。
- `scene.glb` 是预测深度反投影得到的彩色点云，颜色直接来自输入 RGB。
- 模型没有输出 3D Gaussian 参数，也没有 Gaussian rasterization。
- 模型没有渲染新的相机视角或未观测时间点。
- 模型没有使用车辆标注和速度显式移动动态物体。
- 这次运行没有时间插值，也没有尾部外推。

虽然一次前向会联合观察全部 20 张图像，但它仍然是前馈、非自回归模型：跨帧 attention 会让各帧预测互相利用上下文，模型却不会将某一帧预测作为下一帧的新输入，也不会跨调用维护持续更新的全局地图状态。

## 本地实测结果

运行硬件为单张 NVIDIA RTX 4090，官方 1B/512 checkpoint 的结果如下：

- 权重加载时间：43.93 秒（权重位于 HDD）
- 单次 20 帧前向时间：1.087 秒
- 吞吐量：18.40 张/秒
- 峰值显存：7.42 GiB
- 导出点云：约 100 万点
- GLB 内容：1 个点云和 20 个预测相机视锥
- 输出中的深度、相机参数和置信度均为有限数值，没有 NaN/Inf

已有的 Waymo 诊断指标是我们额外编写的检查，不是论文指标：

- 预测相机中心经 Sim(3) 对齐到 Waymo GT 后，ATE RMSE 为 `0.956 m`
- GT 轨迹长度为 `344.877 m`，ATE 约为轨迹长度的 `0.277%`
- 稀疏 LiDAR 深度检查采用逐帧 median scale，AbsRel 为 `0.846`
- 稀疏 LiDAR `delta < 1.25` 为 `29.72%`

相机轨迹结果说明这段前视序列的整体运动恢复较稳定。深度数值明显不理想，但不能把它当作论文指标失败：当前检查使用最近邻缩放的稀疏 LiDAR、逐帧尺度对齐，并且没有复现官方训练时的数据转换、有效像素规则和评测协议。它只能用于发现明显异常，不能用于论文横向比较。

## 论文中的 Waymo 到底怎么使用

VGGT-Omega 论文将 Waymo Open Dataset 列为训练数据来源之一。论文描述的训练数据混合包含公开与内部数据，规模约为 300 万段序列；非合成数据会经过多视图一致性检查和无效深度过滤。

但论文和开源仓库没有公开以下 Waymo 细节：

- Waymo 在训练混合中的采样比例
- 使用哪些 split、相机和时间间隔
- Waymo 原始数据到训练格式的转换脚本
- 动态物体或无效深度的精确掩码规则
- Waymo 专用验证集、指标脚本和论文目标数值

论文公开定量评测使用的是 7 Scenes、NRGBD、ETH3D、DyCheck、Sintel 和 TUM-Dynamic 等数据集，并在相关评测中从场景或序列随机抽取 10 帧。Waymo 不是论文表格中的独立 benchmark。因此，使用 Waymo 可以验证真实自动驾驶长序列上的几何效果和工程可运行性，但不能声称复现了论文 Waymo 指标，因为官方没有提供这样的指标。

## 与 ReconDrive 和 STORM 的区别

| 项目 | ReconDrive | STORM | 当前 VGGT-Omega Waymo 运行 |
|---|---|---|---|
| 每次输入 | 2 个端点时刻 x 6 相机 | 4 个 context 时刻 x 3 相机 | 20 个观测时刻 x 1 前视相机 |
| 时间范围 | 0.5 秒局部窗口 | 约 2 秒局部窗口 | 19 秒，一次联合前向 |
| 核心输出 | 动态 3D Gaussian 和重建 RGB | 场景表示和渲染 RGB | 每个输入时刻的深度、置信度和相机 |
| 中间时刻 RGB | 双端条件插值 | 可包含插值/短期外推 | 不生成 |
| 新视角渲染 | 有 | 有 | 官方推理代码没有 |
| 显式动态建模 | 使用标注、跟踪或运动信息 | 有对应动态建模 | 无标注输入，动态能力隐式学习 |
| 视频组织 | 多个独立窗口拼接 | 多个独立窗口拼接 | 当前是单次 20 帧联合调用 |
| 自回归/结果回灌 | 否 | 否 | 否 |

三者在“前馈、非自回归、不把上一段输出回灌为下一段输入”这一点上相同，但任务定义不同。ReconDrive/STORM 的视频重点是渲染和时间插值/外推；VGGT-Omega 的官方输出重点是从已有 RGB 图像恢复相机与稠密几何。

如果未来为了显存把更长 Waymo 场景切成多个 VGGT-Omega 窗口，那么每个窗口仍会重新读取自己的原始 RGB 并独立前向。官方仓库没有提供跨窗口 Sim(3) 对齐、点云融合或持续地图维护逻辑，这部分需要我们自行实现，不能把独立窗口直接视为一个天然一致的全局重建。

## 当前结论

当前服务器能够稳定使用官方 checkpoint 对 Waymo 图像推理，并得到合理的相机轨迹、深度视频和可交互 3D 点云。现阶段复现的是官方模型的推理能力和输出形式，不是论文定量表格：要复现论文数字，应改用论文列出的公开评测集及其官方协议；Waymo 更适合作为补充的自动驾驶场景定性实验和自定义几何诊断。
