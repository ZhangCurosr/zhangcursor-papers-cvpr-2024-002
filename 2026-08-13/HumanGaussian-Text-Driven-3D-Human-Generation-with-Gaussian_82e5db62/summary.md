---
title: "HumanGaussian-Text-Driven-3D-Human-Generation-with-Gaussian"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_HumanGaussian_Text-Driven_3D_Human_Generation_with_Gaussian_Splatting_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 15:17:36"
field: "文本驱动3D内容生成"
keywords: ["3D Gaussian Splatting", "text-to-3D", "human generation", "score distillation sampling", "neural rendering"]
innovations: ["提出Structure-Aware SDS，通过RGB+Depth双分支联合扩散模型同时优化人体外观与几何", "设计Annealed Negative Prompt Guidance，在低CFG scale下避免过饱和并消除漂浮伪影", "以SMPL-X网格为结构先验初始化3DGS，解决人体文本到3D生成的稀疏梯度和结构不一致问题"]
benchmarks: ["CLIP score", "Aesthetic score", "HPSv2", "User Study (Texture/Geometry/Text Alignment)"]
---

# 论文速读：HumanGaussian: Text-Driven 3D Human Generation with Gaussian Splatting

## 一句话总结
本文提出 **HumanGaussian**，一种将 3D Gaussian Splatting (3DGS) 适配到文本驱动 3D 人体生成任务的高效框架，通过结构感知 SDS（联合优化外观与几何）和退火负提示引导，实现了兼具精细几何与真实外观的高质量 3D 人体生成，单卡 A100 仅需约 1 小时完成优化。

## 研究问题与动机
- 现有文本驱动 3D 人体生成方法多基于 mesh 或 NeRF 表示，mesh 类方法难以刻画配饰、褶皱等细粒度拓扑；NeRF 类方法渲染高分辨率结果耗时长、显存占用大，效率与质量难以兼顾。
- DreamFusion 类 SDS 方法依赖大 CFG scale（如 100）做图像-文本对齐，导致过饱和图案，不利于生成真实人体；同时 SDS 梯度方差高，3DGS 原始自适应密度控制不稳定，会产生模糊和漂浮伪影。
- 将 3DGS 直接用于文本生成面临两个核心挑战：① 3DGS 基于 tile 的光栅化仅对少量高置信度 Gaussian 反向传播，稀疏梯度不利于几何与外观优化，需要结构先验引导；② 初始化问题——SfM 点或通用 Point-E/Shap-E 先验对人体类别不友好，易产生稀疏或不连贯的身体结构。

## 核心贡献（创新点）
1. **提出 HumanGaussian 框架**，首次系统性地将 3DGS 适配到文本驱动 3D 人体生成领域，以 SMPL-X 网格为结构锚点实现高效的高质量人体生成。与之前使用 SfM 点或通用点云先验的工作本质不同，SMPL-X 提供了人体专属的密集且连贯的初始化。
2. **提出 Structure-Aware SDS**，通过训练一个同时去噪 RGB 图像和深度图的 Texture-Structure Joint Model，构建双分支 SDS 损失，联合优化人体外观与几何。与同类工作（如 HumanNorm 微调两个独立的 text-to-normal/depth 模型）相比，本文联合建模避免了深度与法线预测不对齐的问题。
3. **设计 Annealed Negative Prompt Guidance**，将 SDS 分解为 generative score 和 classifier score，仅使用更干净的 classifier score 并以逐步衰减的负提示（negative prompt）替代空条件，在较低 CFG scale（7.5）下即可实现高质量生成，避免过饱和。结合基于 Gaussian size 的 prune-only 阶段消除漂浮伪影。

## 方法详解

### 3.1 预处理与表示
- **SMPL-X 初始化**：从 SMPL-X 网格表面均匀采样 100k 个 3D Gaussian 作为初始中心位置，单位缩放、均值颜色、无旋转。SMPL-X 包含 10,475 顶点和 54 个关键点，比传统 SfM 点更适合人体结构建模。同时将 SMPL-X 关节映射为 COCO 风格 2D 骨架关键点（剔除被遮挡关节）作为 pose condition。
- **3DGS 渲染**：3D Gaussian 由中心位置 $\mu$、协方差 $\Sigma$、颜色 $c$、不透明度 $\alpha$ 参数化，通过 $\alpha$-blending 进行 tile-based 光栅化渲染。

### 3.2 Structure-Aware SDS
- **Texture-Structure Joint Model**：基于 SD 2.0（512 分辨率）微调，采用 v-prediction，同时学习 RGB 和 Depth 的去噪分布。UNet backbone 复制 conv in、首 DownBlock、末 UpBlock、conv out 层分别处理两类目标，pose map $p$ 通过通道拼接作为条件输入。损失函数：
$$\mathbb{E}_{\mathbf{v}, t}\left[\|\mathbf{v}_{\phi}(\mathbf{x}_t; \mathbf{p}, y) - \mathbf{v}_t^{\mathbf{x}}\|_2^2 + \|\mathbf{v}_{\phi}(\mathbf{d}_t; \mathbf{p}, y) - \mathbf{v}_t^{\mathbf{d}}\|_2^2\right]$$
- **双分支 SDS 优化**：渲染得到 RGB 图像 $\mathbf{x}$ 和深度图 $\mathbf{d}$（经 $\alpha$-blending 累加深度值），联合优化：
$$\nabla_\theta \mathcal{L}_{\text{SDS}} = \lambda_1 \cdot \mathbb{E}_{\epsilon_x, t}\left[w_t(\epsilon_\phi(\mathbf{x}_t; \mathbf{p}, y) - \epsilon_x)\frac{\partial \mathbf{x}}{\partial \theta}\right] + \lambda_2 \cdot \mathbb{E}_{\epsilon_d, t}\left[w_t(\epsilon_\phi(\mathbf{d}_t; \mathbf{p}, y) - \epsilon_d)\frac{\partial \mathbf{d}}{\partial \theta}\right]$$
其中 $\lambda_1 = \lambda_2 = 0.5$。

### 3.3 Annealed Negative Prompt Guidance
- **SDS 分解**：将标准 SDS 中的 score matching difference 分解为 generative score $\delta_g = \epsilon_\phi(\mathbf{x}_t; y) - \epsilon$ 和 classifier score $\delta_c = \epsilon_\phi(\mathbf{x}_t; y) - \epsilon_\phi(\mathbf{x}_t; \emptyset)$，即 $\delta = \delta_g + \tau \cdot \delta_c$。
- **退火负提示**：用 negative prompt $y_{neg}$ 替代空条件，定义 negative score $\delta_n = \epsilon_\phi(\mathbf{x}_t; y_{neg}) - \epsilon_\emptyset$，最终损失：
$$\nabla_\theta \mathcal{L}_{\text{SDS}} = \mathbb{E}_{\epsilon, t}\left[w_t\left(\tau_1 \cdot \delta_c - \tau_2 \cdot \delta_n\right)\frac{\partial \mathbf{x}}{\partial \theta}\right]$$
其中 $\tau_1 = 7.5$（DDIM scheduler），$\tau_2$ 从 1.0 逐步衰减至 0.0（在 timestep 200 时降为 0）。该引导同时应用于 RGB 和 Depth 两个分支。
- **Size-Conditioned Gaussian Pruning**：在 densification & pruning 阶段（300–2100 步，间隔 300）之后，增设 prune-only 阶段（2400–3300 步，间隔 300），删除缩放因子超过阈值（0.008）的 Gaussian，以消除低不透明度但大尺寸的漂浮伪影。

### 训练细节
- 总迭代 3600 步，Adam 优化器，学习率：$\mu$ 为 $5 \times 10^{-5}$、$s$ 为 $10^{-3}$、$q$ 为 $10^{-2}$、$c$ 为 $1.25 \times 10^{-2}$、$\alpha$ 为 $10^{-2}$。
- 相机设置：距离 [1.5, 2.0]，fovy [40°, 70°]，elevation [−30°, 30°]，azimuth [−180°, 180°]；1200–3600 步以 25% 概率缩小到头部区域（距离 [0.4, 0.6]）增强面部质量。
- 分辨率 1024，batch size 8，单张 NVIDIA A100 (40GB) 约 1 小时。

## 实验与结果
- **数据集**：深度图标签由 MiDaS 在 LAION 数据集上生成；评估使用从 DreamHuman 随机采样的 30 个 prompt。
- **评估基线**：
  - 通用 text-to-3D：DreamGaussian [77]、GaussianDreamer [89]
  - 专用 text-to-3D human：TADA [41]、DreamHuman [36]
- **定量结果**（Table 2）：
  - CLIP score：**30.82**（最高，超越 TADA 30.13 和 DreamHuman 29.98）
  - Aesthetic score：**6.436**（最高，超越 TADA 5.709）
  - HPSv2：**0.262**（最高，超越 TADA 0.253）
- **用户研究**（Table 1，5 分制）：
  - Texture Quality：**4.24**（vs. TADA 3.76、DreamHuman 3.41）
  - Geometry Quality：**3.88**（vs. TADA 3.53、DreamHuman 3.65）
  - Text Alignment：**4.71**（vs. TADA 4.35、DreamHuman 4.24）
- **结论**：HumanGaussian 在三项主观指标上全面领先，视觉对比显示其在真实外观、连贯身体结构、视角一致性和细粒度细节（如衣物褶皱、耳环）方面优于所有基线。

## 相关工作脉络
1. **DreamFusion [59]**：开山之作，提出 SDS 将 2D 扩散先验蒸馏到 3D 表示（NeRF）；本文将其渲染器替换为 3DGS 并加入结构引导，解决人体建模特有的稀疏梯度问题。
2. **3D Gaussian Splatting (3DGS) [33]**：显式神经表示，支持实时渲染；本文是早期将其引入文本驱动 3D 人体生成领域的尝试，区别于 DreamGaussian [77] 和 GaussianDreamer [89] 等通用 text-to-3D 方法。
3. **DreamHuman [36]**：基于 imGHUM + pose-conditioned NeRF 的文本驱动人体生成；本文使用 3DGS 替代 NeRF 实现更快的渲染和训练速度，同时通过 SMPL-X 初始化改善几何质量。
4. **TADA [41]**：基于 SMPL-X 形变 + UV texture 优化的文本到动画数字人；本文放弃 UV mapping，改用 3DGS 直接在 3D 空间中建模，避免了 UV 展开的限制。
5. **HumanNorm [22]**（同期工作）：同样添加显式结构约束，但微调两个独立的 text-to-normal 和 text-to-depth 模型；本文联合训练单一双分支模型，避免深度与法线预测不对齐问题。
6. **AvatarCLIP [20] / AvatarVerse [94]**：分别基于 NeuS+SMPL 和 ControlNet+DensePose 的人体生成；本文定位更轻量高效，利用 3DGS 的实时渲染优势。

## 局限性与未来方向
- **手部和脚部生成受限**：受限于现有 T2I 模型在手/脚生成上的表现，有时无法忠实渲染这些部位（论文自述）。
- **浮标/漂浮伪影的激进剪枝可能误删有用 Gaussian**：虽然论文认为 SMPL-X 密集初始化可补偿细节损失，但在复杂配饰场景下可能仍有风险。
- **未来方向**：改进手部和脚部的生成质量；探索更鲁棒的密度控制策略。

## 研究启发与可借鉴点
1. **结构先验替代通用先验**：将 SMPL-X 网格作为 3DGS 初始化而非 SfM 点或 Shap-E 点云，这一思路可迁移到其他具有强参数化先验的领域（如动物、人物角色），对解决文本到 3D 中的"Janus 问题"和多面体问题具有借鉴价值。
2. **双模态联合 SDS（RGB + Depth）**：同时利用纹理和几何扩散先验构建双分支 SDS，避免独立微调多个 model 带来的对齐问题；该设计可推广到 normal、segmentation mask 等其他结构模态。
3. **退火负提示引导策略**：将 classifier score 与 annealed negative prompt 结合，在低 CFG scale 下避免过饱和，这一技术已在 concurrent works [32, 91] 中被验证，可复用到其他 text-to-3D 任务。
4. **Size-based Prune-only 阶段**：在 SDS 高方差梯度导致的伪影问题上，通过 Gaussian 尺寸阈值进行后处理剪枝而非依赖梯度信号，思路简洁有效，可用于其他基于 3DGS 的生成任务。
5. **头部区域 zoom-in 策略**：在训练中 25% 概率缩小相机距离以增强面部细节，这一简单的采样策略值得在需要关注局部细节的 3D 生成任务中借鉴。

## 关键术语表
- **Score Distillation Sampling (SDS)**：将预训练 2D 扩散模型的噪声预测梯度蒸馏到 3D 场景表示上的优化技术，是 DreamFusion 的核心机制。
- **3D Gaussian Splatting (3DGS)**：一种显式 3D 场景表示方法，使用各向异性高斯椭球体集合，通过 tile-based 光栅化实现实时渲染。
- **SMPL-X**：包含身体、手部和面部的 3D 参数化人体模型，含 10,475 个顶点和 54 个关键点，本文用作 3DGS 初始化的结构先验。
- **Classifier-Free Guidance (CFG)**：通过引入无条件（空 prompt）和条件预测的差值来增强文本-图像对齐的扩散模型引导技术。
- **Annealed Negative Prompt Guidance**：用负提示替代空条件构建 cleaner classifier score，并以退火方式调整负提示权重以平衡质量与多样性。
- **v-prediction**：扩散模型的一种训练目标，预测 $\mathbf{v} = \alpha_t \epsilon - \sigma_t x$ 而非直接预测噪声 $\epsilon$，有助于稳定训练。
- **Linear Blend Skinning (LBS)**：基于骨骼关节和_blend weights_ 将非刚性变形模板映射到目标姿态的标准蒙皮技术。
- **α-blending**：3DGS 中沿视线方向对排序后的高斯椭球进行透明度累乘的颜色合成方法。

## 可复现要素
- **数据集**：LAION（用于 Joint Model 训练）；深度图由 MiDaS 标注；评估 prompt 来自 DreamHuman。论文未提及评估数据集是否公开。
- **代码/权重**：论文未明确声明开源（项目页面为 https://alvinliu0.github.io/projects/HumanGaussian），但基于 ThreeStudio [14] 和 SD 2.0 实现。
- **关键超参**：Gaussian 初始化数量 100k；总迭代 3600 步；densification & pruning 300–2100 步（间隔 300）；prune-only 2400–3300 步（阈值 0.008）；$\tau_1 = 7.5$，$\tau_2$ 从 1.0 衰减至 0.0（timestep 200）；$\lambda_1 = \lambda_2 = 0.5$；学习率 $\mu$: $5 \times 10^{-5}$, $s$: $10^{-3}$, $q$: $10^{-2}$, $c$: $1.25 \times 10^{-2}$, $\alpha$: $10^{-2}$；分辨率 1024，batch size 8；单卡 A100 约 1 小时。
