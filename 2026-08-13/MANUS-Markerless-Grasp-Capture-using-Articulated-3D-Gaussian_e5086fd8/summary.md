---
title: "MANUS-Markerless-Grasp-Capture-using-Articulated-3D-Gaussian"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Pokhariya_MANUS_Markerless_Grasp_Capture_using_Articulated_3D_Gaussians_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 12:26:16"
field: "手-物体交互与抓取捕捉"
keywords: ["markerless grasp capture", "3D Gaussian splatting", "hand-object interaction", "contact estimation", "multi-view RGB dataset", "articulated hand model"]
innovations: ["基于可关节 3D Gaussian 的手部表示 MANUS-Hand，摒弃模板依赖并支持高效接触估计", "显式 Gaussian 位置拼接实现瞬时/累积接触图计算，无需隐式场密集采样", "发布 50+ 相机多视角数据集 MANUS-Grasps 并引入湿颜料 ground-truth 接触验证协议"]
benchmarks: ["MANUS-Grasps wet-paint evaluation (15 sequences)", "InterHand2.6M for hand rendering quality", "ContactDB / ContactPose / DexYCB / ARCTIC / HOnnotate (comparative context)"]
---

# 论文速读：MANUS-Markerless-Grasp-Capture-using-Articulated-3D-Gaussians

## 一句话总结
本文提出 MANUS，一种基于**可关节 3D Gaussian 表示**的无标记手-物体抓取捕捉方法，通过多视角 RGB 数据直接优化高斯原语来拟合手部形状，并利用高斯位置显式计算接触图；同时发布大规模多视角数据集 MANUS-Grasps（50+ 相机、30+ 场景、~7M 帧），支撑后续研究。

## 研究问题与动机
- 精确估计手-物体接触对机器人、混合现实等应用至关重要，但当前模板方法（MANO / 骨架 / 网格）低维表示无法精确拟合真实手部形状，导致接触估计偏差大。
- 隐式神经场（如 LiveHand、TAVA）在形状与外观上更接近摄影真实感，但接触图需要密集采样评估体积函数，计算成本高，且难以直接拼接手与物体的表征。
- 现有抓取数据集要么依赖特殊传感器/标记（接触 DB / ContactPose / GRAB），要么仅少量视角（H2O、HO3D、DexYCB），无法满足神经场训练所需的数十个视角与 360° 覆盖。

## 核心贡献（创新点）
1. **MANUS-Hand：基于 3D Gaussian Splatting 的可关节手部表示**——通过骨骼驱动 + 线性 blend skinning + 网格权重插值避免模板分配伪影；与 LiveHand/TAVA 相比，无需模板拟合即可逐视角优化，且支持高效接触计算。
2. **基于显式 Gaussian 位置的手-物体接触估计**——在 posed 空间直接计算手/物高斯对间的欧氏距离并设定阈值 τ，生成瞬时接触图与累积接触图；相比隐式场的密集采样方案，计算效率高数个量级。
3. **MANUS-Grasps：大规模多视角 RGB 抓取数据集**——53 台 120 FPS 相机、30+ 日常场景、3 名受试者、~7M 帧，并提供 15 条湿颜料转移的 ground-truth 接触序列用于无偏量化评估。
4. **湿颜料接触验证协议**——利用物体表面绿色湿颜料在抓取后转移到手的视觉残差作为累积接触金标准，为模板/隐式方法提供直观可比的 IoU / F1 评测依据。

## 方法详解

### 4.1 MANUS-Hand
- **骨架与自由度**：21 根骨、26 DoF；使用 AlphaPose [18] + 逆运动学（IK）拟合得到 3D 关节位置与骨长。
- **Gaussian 初始化**：在 canonical 姿势下，每根骨中点为正态分布中心、标准差按骨长缩放；协方差/不透明度/球谐系数按 [29] 协议初始化。
- **变形机制**：canonical → posed 通过前向运动学 + 线性 blend skinning；skin weight 采用 Fast-SNARF [11] 风格的**网格权重（grid weights）**：先在 canonical 空间用最近邻将 MANO skin weight 分配到体素网格，再对查询 Gaussian 做三线性插值得到 W，避免直接分配 MANO weight 导致的训练伪影。
- **变换公式**：
  - 每个 Gaussian 的变换矩阵 $T_g = W T_b$，posed 位置 $\mu_p = T_g \mu$。
  - 协方差 $\Sigma_p = R_g \Sigma R_g^T$，其中 $R_g$ 由 $T_g$ 的旋转部分导出。
  - 颜色查询：将 posed 视角方向 $v_p^g$ 反变换到 canonical 视角 $v_c^g = T_g^{-1} v_p^g$，再查球谐系数得到 RGB。
- **渲染**：$\mathcal{I} = \mathcal{R}(\mu_p, v_c, \Sigma_p, \alpha, \phi)$，使用 [29] 的可微栅格化器。
- **损失函数**：$L_h = \alpha \mathcal{L}_1 + \beta \mathcal{L}_{SSIM} + \gamma \mathcal{L}_{perc} + \delta \mathcal{L}_{iso}$，其中各向同性正则 $\mathcal{L}_{iso} = (\min_s / \max_s - s)^2$，$s = 0.4$，防止极端各向异性高斯产生接触渲染伪影。

### 4.2 MANUS：抓取捕捉与接触估计
- **物体表征**：静态 3D Gaussian，优化过程中通过投影到图像并剔除超出物体 mask 的离群 Gaussian 抑制 floaters。
- **抓取组合**：hand Gaussians $G_h$ 与 object Gaussians $G_o$ 直接拼接 $G_f = \{G_o, G_h\}$，经栅格化一次渲染最终抓取帧；这是 Gaussian 表示天然支持拼接的优势，隐式表示难以实现。
- **接触图**：在 posed 空间，对每个 hand Gaussian 找最近 object Gaussian，若距离 $d(G_h, G_o) < \tau$ 则视为接触，输出接触距离图：
  $$C = \begin{cases} d(G_h, G_o), & \text{if } d < \tau \\ 0, & \text{otherwise} \end{cases}$$
  - 瞬时接触图：单帧接触。
  - 累积接触图：$C_{acc} = C_{acc}^{prev} + C^{curr}$，贯穿整个抓取序列。
- 渲染接触图时以接触距离作为每个 Gaussian 的"颜色值"调用公式 (1)。

### 4.3 MANUS-Grasps 数据集
- **采集系统**：53 台 RGB 相机（每面 9 台）均匀分布在立方体体积内，120 FPS、1280×720，帧间错位 ≤ 3 ms；LED 均匀照明；COLMAP 标定内参/外参。
- **采集协议**：① 主体右手自由活动序列；② 仅拍物体；③ 固定物体后拍手抓取；④ 评估序列末端补充 canonical pose。
- **Ground-truth 接触**：15 条评估序列使用绿色湿颜料涂于物体，抓取后颜料转移到手作为物理真值；明亮绿色便于自动分割。
- **标注**：AlphaPose + 3D 三角测量 + IK 得到 2D/3D 关节；1C Filter [8] 做时序平滑；SAM [30] + Instant-NGP [44] 提取 multi-view 一致的 binary mask。

## 实验与结果
- **数据集与基线**：MANUS-Grasps 15 条湿颜料序列；对比 MANO fitting、HARP [28] 与本方法 MANUS。所有方法统一阈值 τ = 0.004；MANO/HARP 网格从 778 顶点细分到 49000 顶点保证公平。
- **接触评估指标**：IoU 与 F1-score（基于 10 个视角渲染的 2D 接触 mask）。
- **主要定量结果**（表 2）：
  - mIoU：Subject1 0.206 / Subject2 0.152 / Subject3 0.275；对手 MANO（0.161/0.135/0.208）与 HARP（0.173/0.148/0.224）均稳定领先，Subject3 提升约 +0.051（vs HARP）。
  - F1：Subject1 0.335 / Subject2 0.251 / Subject3 0.424；对手 MANO（0.270/0.228/0.338）与 HARP（0.289/0.247/0.361）。
- **视图消融**（表 4）：视角数从 30+ 降至 5 时，mFoU/F1 显著下降（如 Subject1 mIoU 0.206 → 0.147），印证多视角缓解自遮挡的假设。
- **手模型视觉质量**（表 3，InterHand2.6M）：PSNR 26.32 / SSIM 0.9872 / LPIPS 0.068，略低于 LiveHand（31.16），但显著高于 TAVA（22.85）；论文强调本方法首要目标是接触精度而非摄影级真实感。

## 相关工作脉络
- **MANO / HARP**：模板-based 手部参数模型；本工作摒弃模板假设，用 Gaussian 原语直接拟合个体差异形状，且天然支持接触度量。
- **LiveHand / TAVA**：基于隐式神经场的手部表示；本工作改用 3D Gaussian 获得更快的训练/推理速度与更简洁的接触计算，代价是放弃环境光遮蔽与阴影，不追求 photorealism。
- **ContactDB / ContactPose**：基于热成像 / RGB-D 的多视角抓取数据集，但视角数量有限（3-16）、物体非真实、缺少神经场所需的高密度多视角；MANUS-Grasps 以 50+ 相机弥补这一缺口。
- **HOnnotate / ARCTIC / DexYCB**：markerless 抓取数据集，但视角数少或聚焦双手操作；MANUS-Grasps 首次面向神经场训练提供 360° 密集覆盖。
- **Dynamic 3D Gaussians（4DGS、Dynamic 3D Gaussians [67]）**：允许 Gaussian 自由运动/形变，但缺乏基于骨骼的低维可控性；MANUS-Hand 通过骨架+skin weight 实现可控 articulation，更适合抓取捕捉任务。
- **Phys-sim grasp methods（D-Grasp、ArtiGrasp）**：限于合成抓取；MANUS 面向真实 RGB 数据且端到端可训练。

## 局限性与未来方向
- 当前方法仅建模**单手抓静态物体**，未处理双手协同、长时程操控及皮肤拉伸引起的非线性形变。
- 数据集采集需要 53 台同步高速相机与立方体体积，硬件门槛高、普及性受限（论文已开源数据缓解部分问题）。
- 放弃环境光遮蔽与阴影使视觉质量逊于 LiveHand，限制直接用于 photorealistic VR/AR。
- 接触评估指标（IoU/F1 on 2D 投影 mask）可能受视角选择与阈值 τ 影响，论文承认有改进空间。
- 未来可扩展到双手交互、动态物体、更长序列、以及引入阴影/遮挡建模。

## 研究启发与可借鉴点
1. **网格权重插值（grid weights）替代模板 skin weight 直赋**：对任意基于 Gaussian 的形变表征，该策略有效避免训练过程中 skin weight 漂移造成的伪影，可作为通用技巧迁移到人体/动物 articulated Gaussian 模型。
2. **Gaussian 原语拼接实现接触估计**：手与物体的 Gaussian 直接 concat 即可一次性栅格化并计算接触距离，无需额外体积查询或 ray marching，为其他 hand-object 交互工作提供低成本的接触管线设计范式。
3. **湿颜料 gold-standard 接触验证**：用物理颜料转移获取无需标注的接触真值，为未来 hand-object 接触评测提供一种可复用、低成本、跨方法的客观基准。
4. **视角数量-接触精度定量关系**：表 4 清晰展示 30+ 视角对接触质量的必要性，提示后续方法设计应把"多视角覆盖"作为前置条件而非可选项。
5. **各向同性正则 $\mathcal{L}_{iso}$**：控制 Gaussian 尺度比防止极端拉长的原语，对任何需要基于原语距离进行接触/碰撞评估的任务均具有参考价值。

## 关键术语表
- **MANUS-Hand**：基于 3D Gaussian Splatting 的可关节手部表示，通过骨架驱动与网格权重插值实现模板无关的形状与外观拟合。
- **3D Gaussian Splatting**：用各向异性 3D 高斯原语表示场景，通过可微栅格化实现高效渲染与优化。
- **Linear Blend Skinning (LBS)**：基于骨骼变换矩阵的加权平均实现对顶点的形变；本文扩展至 Gaussian 中心与协方差。
- **Grid Weights**：先在 canonical 空间构建体素网格并分配 skin weight，再通过三线性插值获取查询 Gaussian 的权重，避免 MANO weight 直赋的伪影。
- **Instantaneous vs. Accumulated Contact Map**：前者为单帧手-物接触距离图，后者为整段抓取序列接触状态的累加视图。
- **Wet-Paint Ground Truth**：在物体表面涂抹绿色湿颜料，抓取后颜料转移到手，作为无需人工标注的接触真值。
- **τ (接触阈值)**：判断两 Gaussian 中心是否接触的距离门限，本文统一设为 0.004。
- **Isotropic Regularizer $\mathcal{L}_{iso}$**：惩罚高斯尺度最小/最大值比偏离 0.4，防止各向异性过度导致的接触伪影。

## 可复现要素
- **数据集**：MANUS-Grasps，论文声明将公开（链接 ivl.cs.brown.edu/research/manus）。
- **代码/权重**：论文未明确开源代码与权重（正文仅给出项目网页），需进一步确认 supplementary 与 GitHub 仓库。
- **关键超参**：接触阈值 τ = 0.004；各向同性正则目标 s = 0.4；L1 / SSIM / perceptual / iso 损失权重论文未列明具体数值（可查阅 supplementary）。
- **训练数据规模**：每 Subject 约 20 个手部姿态序列；评估序列 15 条（湿颜料）。
- **推理加速**：使用 [29] 的 tile rasterizer，单帧渲染时间未见详细报告。
