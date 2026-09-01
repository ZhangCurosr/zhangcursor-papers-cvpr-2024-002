---
title: "HEAL-SWIN-A-Vision-Transformer-On-The-Sphere"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Carlsson_HEAL-SWIN_A_Vision_Transformer_On_The_Sphere_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 12:22:55"
field: "球面计算机视觉"
keywords: ["HEALPix", "Vision Transformer", "球面深度学习", "鱼眼图像", "语义分割", "深度估计", "SWIN Transformer", "自动驾驶"]
innovations: ["将 HEALPix 等面积嵌套网格与 SWIN Transformer 的层次结构系统性结合，实现高效球面 attention", "提出网格平移与螺旋平移两种球面窗口移位策略，螺旋平移在消融中更优", "首次将自动驾驶鱼眼图像视为无畸变球面信号，在 SynWoodScape 和 WoodScape 上全面超越平面 SWIN"]
benchmarks: ["WoodScape", "SynWoodScape", "Stanford 2D-3D-S", "spherical MNIST"]
---

# 论文速读：HEAL-SWIN: A Vision Transformer On The Sphere

## 一句话总结
本文提出了 **HEAL-SWIN**，将天体物理中广泛使用的 HEALPix 球面网格与 SWIN Transformer 相结合，首次将自动驾驶鱼眼图像视为无畸变的球面信号进行端到端训练，在语义分割和深度估计任务上全面超越传统平面 SWIN 基线。

## 研究问题与动机
- **鱼眼图像的投影畸变难题**：高分辨率广角鱼眼图像存在强烈的空间非均匀畸变，传统 CNN/ViT 将其投影到平面网格时会引入不可忽略的几何失真，在自动驾驶等安全关键应用中尤为突出。
- **既有球面方法的分辨率瓶颈**：基于 Driscoll–Healy 网格和傅里叶变换的球面 CNN（如 SphCNN、Spin-SphCNN）采样不均匀（极点密集）、傅里叶变换复杂度随带宽立方增长、且输入仅覆盖半球时卷积定义需额外调整，难以处理高分辨率数据。
- **缺乏球面表示的自动驾驶数据集**：主流自动驾驶/计算机视觉数据集不含鱼眼图像，现有方法只能在对畸变进行预处理或变形卷积核后使用平面网络，无法充分利用球面几何信息。
- **半球数据的利用效率低**：使用全球网格的球面模型对仅覆盖约半球的鱼眼图像存在大量无效计算，浪费算力与显存。

## 核心贡献（创新点）
- **HEAL-SWIN 架构**：首次将 HEALPix 等面积嵌套网格与 SWIN Transformer 的系统性结合，利用两者共有的层次结构实现球面上的 patching、windowing、patch merging/expansion 操作，计算开销极小。
- **两种球面窗口平移策略**：提出网格平移（沿基四边形轴）和螺旋平移（沿纬度环从极点到极点）两种窗口移位方案，其中螺旋平移在消融实验中表现更优，有效解决全局信息分布问题。
- **半球自适应 HEALPix 子网格**：仅使用 12 个基础四边形中的 8 个（"base pixels"），精确覆盖约半球，避免全球网格的无效计算，同时保持球面几何完整性。
- **自动驾驶鱼眼图像的球面范式革新**：首次将汽车应用中的鱼眼图像作为无畸变球面信号处理，在 SynWoodScape 和 WoodScape 两个数据集上证明了球面表示对深度估计（Chamfer 距离）和语义分割（mIoU）的增益。
- **Stanford 2D-3D-S 室内球面 benchmark 新 SOTA**：在 Stanford 2D-3D-S 室内鱼眼分割任务上以 44.3 mIoU / 61.9 mAcc 超越 Gauge CNN、UGSCNN、HexRUNet 等同类球面模型。

## 方法详解

### 3.1 SWIN Transformer 回顾
SWIN Transformer 将输入图像划分为 patch，在局部窗口内计算自注意力，每两层窗口沿两个轴平移半个窗口大小以实现跨窗口信息流通；通过 patch merging 降采样、patch expansion 上采样，配合 skip connection 构成 UNet 状架构。

### 3.2 HEALPix 网格
HEALPix（Hierarchical Equal Area iso-Latitude Pixelation）由 12 个等面积四边形铺满球面，沿边进行 $n_{\text{side}}$ 次细分（$n_{\text{side}}$ 为 2 的幂），生成 $n_{\text{pix}} = 12 \cdot n_{\text{side}}^2$ 个等面积四边形像素。像素中心位置按嵌套顺序（nested ordering）或环顺序（ring ordering）排列。本文使用 Python 包 `healpy` 进行索引转换和插值。

**半球适配**：仅取前 8/12 个基础四边形（"base pixels"），覆盖约半球，对应嵌套排序的前 8/12 项。

### 3.3 HEAL-SWIN 设计

**Patching & Windows**：输入为嵌套排序的一维像素列表，将连续 $n_{\text{patch}}$（4 的幂）个像素合并为一个 patch，再将连续 $n_{\text{win}}$（4 的幂）个 patch 划分为一个 window 进行注意力计算。Patch merging/expansion 同样以 4 的幂合并/展开连续 patch。

**Grid Shifting（网格平移）**：沿基四边形轴的二维方向各平移半个窗口大小，等价于原始 SWIN 的直接推广。在 base pixel 边界处会产生重叠/碰撞，通过重新排列像素并施加 masked attention（对不同来源区域的像素对屏蔽）处理。

**Spiral Shifting（螺旋平移）**：将嵌套顺序转换为环顺序，对列表做长度为 $n_{\text{shift}} = \sqrt{n_{\text{win}}}/2$ 的循环移位，再转回嵌套顺序。窗口沿方位角平移，极点处有轻微形变（向极点方向形变增大），但图像内部无边界效应。消融实验表明螺旋平移略优于网格平移。

**Relative Position Bias（相对位置偏置）**：
$$
\text{Att}(Q,K,V) = \text{SoftMax}(QK^\top/\sqrt{d} + B)V
$$
$$
B_{ij} = \hat{B}_{x(i)-x(j)+\sqrt{n_{\text{win}}},\; y(i)-y(j)+\sqrt{n_{\text{win}}}}
$$
其中 $(x(i), y(i))$ 为窗口内像素 $i$ 的笛卡尔坐标。$\hat{B} \in \mathbb{R}^{(2\sqrt{n_{\text{win}}}-1)\times(2\sqrt{n_{\text{win}}}-1)}$ 为可学习矩阵，在所有 window 间共享。实验表明绝对位置嵌入无性能收益。

**损失函数**：语义分割使用加权交叉熵；深度估计使用 $L_2$ 损失，天空区域在训练和评估时 mask 掉；深度图标准化为零均值单位方差；重采样使用最近邻插值以保持高对比度边缘。

## 实验与结果

### 数据集
- **WoodScape**：8234 张真实鱼眼街景图像（美、欧、中国），20 类语义分割。
- **SynWoodScape Large**：8 类（大面积目标），12 类 SubsetData Large+AD（含行人、信号灯、交通标志等细粒度类）。
- **Stanford 2D-3D-S**：1413 张室内 RGB-D 鱼眼图像，14 类语义分割。

### 语义分割结果（Table 1）
| 模型 | 数据集 | mIoU |
|------|--------|------|
| HEAL-SWIN | Large SynWoodScape | **0.947** |
| SWIN | Large SynWoodScape | 0.918 |
| HEAL-SWIN | Large+AD SynWoodScape | **0.841** |
| SWIN | Large+AD SynWoodScape | 0.809 |
| HEAL-SWIN | WoodScape | **0.628** |
| SWIN | WoodScape | 0.617 |

→ 在所有三个数据集上 HEAL-SWIN 均优于平面 SWIN，最大提升 **+29 mIoU**（Large+AD SynWoodScape）。

### Stanford 2D-3D-S 球面模型对比（Table 3）
| 模型 | mIoU | mAcc |
|------|------|------|
| HexRUNet | 43.3 | 58.6 |
| Spin-SphCNN | 41.9 | 55.6 |
| **HEAL-SWIN（本文）** | **44.3** | **61.9** |

→ 以 1.5M 参数量超越所有同类球面模型。

### 深度估计（Figure 2 左）
在 SynWoodScape 上用 Chamfer 距离评估预测点云与 GT 点云的相似度，HEAL-SWIN 显著优于 SWIN，证明球面表示学到了更好的 3D 几何表示。

### 推理速度（Table 2）
在 A40 GPU 上，8×256² 分辨率下 HEAL-SWIN 与 SWIN 推理时间几乎相同（297 ns/px vs 296 ns/px）；更低分辨率下 HEAL-SWIN 因内存布局优势反而更快。

### 数据集规模消融（Figure 6）
随着训练数据量增加，HEAL-SWIN 相对于 SWIN 的性能优势持续扩大，表明球面表示更好地利用了大数据。

### 超参消融（Appendix Table 9）
最优配置：$n_{\text{win}}=16$，$n_{\text{patch}}=4$，$n_{\text{shift}}=2$，螺旋平移。

## 相关工作脉络
- **Spherical Transformer [7]**：使用多种球面网格提取 patch 后应用 ViT，但未引入局部注意力和窗口平移机制，难以处理高分辨率图像；HEAL-SWIN 在此基础上加入 SWIN 的 windowed attention 和 shifting。
- **Climformer [4]**：将二十面体网格与 AFNO（Adaptive Fourier Neural Operator）结合用于气候预测；HEAL-SWIN 使用 HEALPix 等面积网格并采用注意力而非傅里叶变换，避免傅里叶变换的高计算复杂度。
- **SphCNN / Spin-SphCNN [18, 19]**：基于 Driscoll–Healy 网格的傅里叶球面 CNN，采样不均匀且受限于傅里叶域立方复杂度；HEAL-SWIN 不需要傅里叶变换，直接在工作空间处理高分辨率数据。
- **DeepSphere [11, 41]**：基于图的旋转等变球面 CNN，使用非均匀采样；HEAL-SWIN 利用 HEALPix 的嵌套结构进行层次化窗口注意力，能更高效地处理高分辨率且仅覆盖半球的输入。
- **SWPT [25]**：将点云投影到球面并用旋转实现 window shifting；HEAL-SWIN 处理原生球面数据，利用 HEALPix 网格的结构特性设计平移策略，无需显式旋转。
- **HexRUNet [50]**：基于二十面体网格采样的球面 CNN，用于 Stanford 2D-3D-S 分割；HEAL-SWIN 在相同 benchmark 上达到更高 mIoU（44.3 vs 43.3）。

## 局限性与未来方向
- **半球边界的网格裁剪浪费**：当前使用 8/12 个基础四边形覆盖半球，边界处仍有部分图像像素未被覆盖而部分网格区域闲置；可通过在嵌套结构中下探边界并适配平移策略改善。
- **相对位置偏置未区分极区与赤道区**：当前 $\hat{B}$ 在所有 base quadrilateral 间共享，未考虑极点附近和赤道附近的几何差异；可从网格结构推导校正项。
- **UNet-like 架构非最新 SOTA**：基于 SWIN-Unet 的设计在语义分割任务上并非最先进；引入现代 Vision Transformer Decoder Head（如 Segmenter、Mask2Former 风格）到 HEALPix 可能进一步提升性能。
- **缺乏旋转等变性**：当前模型对球面旋转不等变；对于语义分割、深度估计等任务，引入等变性（参考 [21]）可能带来显著性能提升；局部变换等变性也是一个有趣方向。
- **相对位置嵌入实验**：仅测试了相对位置偏置，绝对位置嵌入无收益但未深入探索混合策略。

## 研究启发与可借鉴点
- **网格结构与注意力机制的层次对齐**：HEALPix 的嵌套结构与 SWIN 的 patch merging/windowing 天然契合，这一"网格层次对齐注意力结构"的设计思路可迁移到其他非欧网格（如多面体网格、各向异性网格）上的 Transformer 设计。
- **球面数据处理的傅里叶自由范式**：证明无需傅里叶变换也能高效处理球面数据，为高分辨率球面深度学习开辟了新路径，尤其适合仅覆盖部分球面的输入（如鱼眼相机、卫星部分覆盖图像）。
- **半球的网格子集策略**：仅使用 8/12 基础四边形的做法对任何"部分球面覆盖"场景（如天空成像、地平线以上传感器）均有参考价值，避免全球网格的算力浪费。
- **Chamfer 距离评估深度估计的 3D 质量**：将深度图转换为球面点云并用 Chamfer 距离评估，直接衡量下游 3D 任务所需的几何精度，这一评估范式值得在自动驾驶深度估计研究中推广。
- **数据量放大时的性能增益趋势**：HEAL-SWIN 在大数据量下优势扩大，提示球面表示的容量/归纳偏置更适合大规模训练，可与本团队的大模型/数据规模策略结合。

## 关键术语表
- **HEALPix**（Hierarchical Equal Area iso-Latitude Pixelation）：一种将球面划分为等面积、层次嵌套四边形的球面网格系统，源于宇宙微波背景辐射观测。
- **SWIN Transformer**：通过局部窗口自注意力加跨层窗口平移实现高效 vision transformer，支持层次化特征金字塔。
- **Base pixels**：HEALPix 网格的 12 个初始等面积四边形，本文选取其中 8 个覆盖约半球。
- **Nested ordering**：HEALPix 像素按从粗到细的四边形细分顺序排列的一维索引方式。
- **Ring ordering**：HEALPix 像素按等纬度环从一极到另一极排列的索引方式。
- **Grid shifting**：沿 HEALPix 基四边形轴方向的窗口平移策略，在 base pixel 边界处有碰撞效应。
- **Spiral shifting**：沿纬度环从极点到极点的窗口平移策略，内部无边界效应，消融实验中表现更优。
- **Chamfer distance**：衡量两个点云之间相互最近距离的平均值，越低表示预测点云越接近 GT，用于深度估计的 3D 质量评估。

## 可复现要素
- **数据集**：WoodScape（公开）、SynWoodScape（公开）、Stanford 2D-3D-S（公开）、spherical MNIST（自建）。
- **代码**：论文声明代码已公开（官网 footnote 1）。
- **权重**：未提及预训练权重开源情况。
- **关键超参**：
  - 分割任务（汽车）：$n_{\text{patch}}=4$，$n_{\text{win}}=64$，螺旋平移 $n_{\text{shift}}=4$，约 41M 参数，SWIN-UNet 架构。
  - 分割任务（室内 Stanford）：$n_{\text{side}}=64$（49k 像素），$n_{\text{win}}=16$，$n_{\text{patch}}=4$，$n_{\text{shift}}=2$，约 1.5M 参数。
  - 深度估计：同分割架构，输出通道数=1，$L_2$ loss，天空区域 mask。
- **硬件**：A40 GPU。
- **依赖库**：`healpy`（Python 球面像素化与球谐变换库）。
