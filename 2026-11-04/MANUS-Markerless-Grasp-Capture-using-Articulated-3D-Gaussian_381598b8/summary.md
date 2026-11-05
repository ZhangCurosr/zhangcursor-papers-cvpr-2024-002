---
title: "MANUS-Markerless-Grasp-Capture-using-Articulated-3D-Gaussian"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Pokhariya_MANUS_Markerless_Grasp_Capture_using_Articulated_3D_Gaussians_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 00:42:39"
field: "手物交互与接触估计"
keywords: ["hand-object grasp capture", "3D Gaussian splatting", "articulated hand modeling", "contact estimation", "multi-view reconstruction", "markerless motion capture"]
innovations: ["将3D Gaussian Splatting扩展至articulated手部，实现高效接触估计", "提出网格权重skin weighting技巧解决Gaussian蒙皮训练不匹配问题", "构建50+相机7M帧多视角手物抓取数据集并提供湿油漆接触真值评估"]
benchmarks: ["MANUS-Grasps (湿油漆接触评估)", "InterHand2.6M (视觉质量评测)"]
---

# 论文速读：MANUS-Markerless-Grasp-Capture-using-Articulated-3D-Gaussians

## 一句话总结
本文提出 MANUS，一种基于可 articulated 3D Gaussian 表示的无标记手物抓取捕捉方法，通过多视角（50+ 相机）RGB 视频实现高精度手形重建与接触区域估计；同时构建了包含 700 万帧的大规模多视角数据集 MANUS-Grasps，并提供基于湿油漆转印的接触评估基准。

## 研究问题与动机
1. **手物接触建模困难**：抓取捕捉的核心挑战是准确估计手与物体的接触区域，但现有方法（骨骼、网格、参数化模型）难以精确表示手部形状，导致接触估计误差大。
2. **隐式表示效率低**：近期的 articulated neural implicit representations（如 LiveHand、Lisa）虽能精确建模形状，但接触估计需昂贵采样，计算成本高。
3. **数据集视角不足**：现有数据集要么使用特殊硬件（热成像、标记手套），要么只有少数视角（3-9 个），自遮挡严重，无法支持神经场类方法。
4. **缺乏接触真值**：大多数数据集仅提供手部位姿，缺乏准确的接触标注；热成像（ContactDB）虽提供接触信息，但存在热量散逸问题。

## 核心贡献（创新点）
1. **MANUS-Hand 表示**：将 3D Gaussian Splatting 扩展至可驱动手部，使用骨骼+线性blend skinnning控制高斯位置，无需模板即可生成个性化手形；与 MANO/TAVA 等参数化或隐式方法相比，原生支持高效接触计算。
2. **MANUS 抓取捕捉框架**：将可动 MANUS-Hand 与静态物体高斯表示拼接后联合渲染，直接基于 3D 空间高斯位置邻近性计算瞬时与累积接触图；与隐式方法相比，免除了体积采样开销。
3. **湿油漆接触评估方法**：在 15 条序列中使用湿绿色油漆转印作为接触真值，利用颜色自动分割提取 Gold Standard 接触图，无需额外设备即可客观评估接触精度。
4. **MANUS-Grasps 数据集**：构建 7M+ 帧、50+ 相机、30+ 场景、3 个被试的多视角 RGB 数据集，首次为神经场方法提供充足视角覆盖。
5. **网格权重 Skin Weighting**：借鉴 Fast-SNARF 的 canonical grid，用三线性插值分配 skin weight，避免了直接复制 MANO 权重导致的训练不稳定和伪影。

## 方法详解

**MANUS-Hand 表示**
- 骨架：21 根骨骼、26 DOF，通过 AlphaPose + IK 估计初始关节位置。
- 高斯初始化：在 canonical 姿态下，每个骨骼中点附近按正态分布放置 Gaussian，标准差与骨骼长度匹配；协方差、opacity、SH 系数按 3DGS 协议初始化。
- 网格权重（Grid Weights）：构建 canonical 空间网格，用最近邻法分配 MANO 风格的 skin weight 到体素；查询时三线性插值得到每个 Gaussian 的权重 $W$。
- 正向运动学 + 线性 blend skinnning：$T_g = W T_b$，$\mu_p = T_g \mu$，$\Sigma_p = R_g \Sigma R_g^T$。
- 外观：SH 系数在 canonical 空间优化，渲染时通过 $T_g^{-1}$ 将 posed view direction 映射回 canonical space 查询颜色。
- 损失函数：$\mathcal{L}_h = \alpha \mathcal{L}_1 + \beta \mathcal{L}_{SSIM} + \gamma \mathcal{L}_{perc} + \delta \mathcal{L}_{iso}$，其中各向同性正则化 $\mathcal{L}_{iso} = (\frac{\min_s}{\max_s} - s)^2$，$s=0.4$。

**MANUS 抓取捕捉**
- 物体表示：静态 Gaussian 拼接，剪枝 outlier 防止 floaters。
- 双手拼接：$G_f = \{G_o, G_h\}$，利用 3DGS rasterizer 自然合并。
- 接触估计：对每个 hand Gaussian 找最近 object Gaussian，距离 $d(G_h, G_o) < \tau$ 则视为接触；瞬时接触图 $C$ 按此阈值二值化，累积接触图 $C_{acc}$ 对所有帧累加；渲染时将接触距离作为颜色值输出。

**MANUS-Grasps 数据采集**
- 53 台 RGB 相机（每面 9 台），120 FPS，1280×720，3ms 同步误差。
- 校准：COLMAP + 墙面 fiducial markers。
- 湿油漆序列：物体涂亮绿色油漆，抓取后还原 canonical 姿态，自动分割接触印记作为 ground truth。

## 实验与结果

**MANUS-Hand 视觉质量（InterHand2.6M）**

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | Test(s)↓ |
|------|-------|-------|--------|---------|
| TAVA | 22.85 | 0.983 | 0.099 | 11.00 |
| LiveHand | 31.16 | 0.9818 | 0.0278 | 0.022 |
| Ours | 26.32 | 0.9872 | 0.068 | 0.049 |

注：MANUS-Hand 非专为 photorealism 设计（无环境光遮蔽/阴影），但视觉质量优于 TAVA，接近 LiveHand。

**抓取接触评估（MANUS-Grasps 湿油漆序列，mIoU / F1）**

| 方法 | Subject1 mIoU | Subject2 mIoU | Subject3 mIoU |
|------|-------------|-------------|-------------|
| MANO | 0.161 | 0.135 | 0.208 |
| HARP | 0.173 | 0.148 | 0.224 |
| **Ours** | **0.206** | **0.152** | **0.275** |

F1 score 最佳：Subject1=0.335，Subject2=0.251，Subject3=0.424（均为 Ours 最优）。

**视角数量消融（Table 4）**：随相机数从 30+ 降至 5，mIoU 和 F1 显著下降，验证密集多视角对抑制自遮挡、提升接触估计精度的重要性。

## 相关工作脉络

1. **MANO（模板参数化模型）**：广泛使用的手部参数化模型，形状维度低，接触估计易因 mesh 与真实形貌偏差导致过分割（论文 Fig.1 直观展示）。
2. **HARP（单目个性化手部重建）**：基于隐式神经场的单目手部重建，需大量采样计算接触，效率低；MANUS 使用显式 Gaussian 消除此开销。
3. **LiveHand / TAVA（articulated neural implicit representations）**：当前 SOTA 手部显式/隐式表示，视觉质量高但接触估算困难；MANUS 以可接受的轻微画质妥协换取高效的接触计算能力。
4. **ContactDB / ContactPose（接触数据集）**：分别使用热成像和 RGB-D 提供接触标注，但视角有限或非真实物体；MANUS-Grasps 提供 50+ 视角 RGB 数据与物理接触真值。
5. **3D Gaussian Splatting（Kerbl et al.）**：静态场景高保真渲染基础；本文将其扩展至 articulated 结构，是关键技术前置。
6. **Fast-SNARF**：提出 canonical grid + trilinear interpolation 的 skin weighting 方案，本文借鉴此技巧解决 Gaussian 训练过程中的权重不匹配问题。

## 局限性与未来方向

1. **仅支持单手持握静态物体**，未处理双手交互或工具间接抓取。
2. **未建模皮肤拉伸等 pose-dependent 非线性形变**，在极端姿态下可能产生几何失真。
3. **长时间手物操作未涉及**，现有方法针对单次抓取瞬间。
4. **采集系统昂贵且不可普及**：53 台高速相机阵列限制了方法的可及性；作者承诺公开数据集以缓解此问题。
5. **接触评估指标有改进空间**：mIoU/F1 对接触边界敏感，可能存在评估噪声。

## 研究启发与可借鉴点

1. **网格权重 Skin Weighting 技巧**：借鉴 Fast-SNARF 的 canonical grid + 三线性插值，可有效解决 Gaussian 化表示中的皮肤权重分配问题，适用于任何基于 Gaussian 的 articulated body 建模工作。
2. **显式 Gaussian 拼接替代隐式场融合**：将 hand 和 object 的高斯集合直接 concat 后统一 rasterize，天然支持接触距离计算，为其他 hand-object 交互任务提供了简洁的接触估计范式。
3. **湿油漆转印评估范式**：将物理接触痕迹转化为可自动分割的视觉信号，为接触估计提供了无需额外标注的真值获取方法，可迁移至其他接触相关任务。
4. **各向同性正则化 $\mathcal{L}_{iso}$**：限制高斯各向异性程度，既提升接触渲染质量又防止训练伪影，是 Gaussian-based 几何表示的一个实用正则化技巧。
5. **密集多视角对接触估计的决定性作用**：实验定量验证了视角数量与接触精度的单调关系，提示后续研究应优先保障采集视角密度而非单纯增加模型复杂度。

## 关键术语表

**3D Gaussian Splatting**：一种基于 3D 高斯原语显式表示的场景渲染方法，通过可微分 rasterizer 实现高效训练与实时渲染。

**MANUS-Hand**：本文提出的无模板可驱动手部模型，由 21 根骨骼驱动的 3D Gaussian 集合构成。

**Linear Blend Skinning (LBS)**：将骨骼变换矩阵按权重线性混合以驱动顶点（或 Gaussian）运动的经典蒙皮技术。

**Grid Weights**：将 skin weight 预分配至 canonical 空间体素网格，再通过三线性插值查询每个 Gaussian 权重的技巧。

**Instantaneous Contact Map**：单帧中 hand 与 object Gaussian 间满足距离阈值的接触二元图。

**Accumulated Contact Map**：整个抓取过程中所有帧瞬时接触图的时序累加，近似真实接触面积。

**Wet Paint Transfer**：利用物体表面湿油漆在抓取后残留在手上的痕迹作为接触区域的物理真值。

**Isotropic Regularizer**：约束高斯各向同性程度的正则项，防止接触渲染时出现异常拉伸伪影。

## 可复现要素

- **数据集**：MANUS-Grasps（论文声明将公开，含 7M+ 帧、50+ 相机、15 条湿油漆评估序列、ground truth 相机位姿、分割 mask、接触标注）
- **代码/权重**：项目页面 ivl.cs.brown.edu/research/manus（论文未明确说明 GitHub 开源状态）
- **关键超参**：接触距离阈值 $\tau = 0.004$；各向同性正则化目标比值 $s = 0.39$（论文写 0.4）；训练使用约 20 个手部姿态序列；网格权重初始化基于 MANO 权重。
