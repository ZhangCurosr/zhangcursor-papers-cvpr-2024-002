---
title: "HumanGaussian-Text-Driven-3D-Human-Generation-with-Gaussian"
source: https://openaccess.thecvf.com//content/CVPR2024/papers/Liu_HumanGaussian_Text-Driven_3D_Human_Generation_with_Gaussian_Splatting_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 12:24:16"
field: "文本驱动3D人体生成"
keywords: ["3D Gaussian Splatting", "Text-to-3D", "Human Generation", "Score Distillation Sampling", "Diffusion Model"]
innovations: ["提出Structure-Aware SDS联合优化人体外观与几何", "设计退火负提示引导解决SDS过饱和问题", "基于SMPL-X先验初始化高斯中心并引入尺寸条件剪枝消除漂浮伪影"]
benchmarks: ["CLIP Score", "Aesthetic Score", "HPSv2", "User Study (Texture/Geometry/Text Alignment)"]
---

# 论文速读：HumanGaussian-Text-Driven-3D-Human-Generation-with-Gaussian

## 一句话总结
本文提出 **HumanGaussian**，一种基于 3D Gaussian Splatting 的高效文本驱动 3D 人体生成框架，通过结构感知 SDS（Structure-Aware SDS）联合优化外观与几何，并引入退火负提示引导解决过饱和与浮动伪影问题，实现高质量、细粒度的人体生成。

## 研究问题与动机
1. **现有 3D 人体生成方法质量与效率难以兼得**：基于 mesh 的方法（如 TADA）难以建模细粒度拓扑（配饰、皱纹）；基于 NeRF 的方法（如 DreamHuman）训练与渲染耗时耗显存。
2. **3DGS 直接迁移到文本驱动人体生成面临两大挑战**：① 原始 3DGS 依赖 SfM 点初始化，在人体领域容易因点云过稀疏或结构不连贯导致多脸 Janus 问题；② 朴素 SDS 需要极大 CFG 尺度（如 100）才能保证图文对齐，导致生成结果过饱和失真。
3. **结构先验利用不足**：既有工作多依赖通用先验（如 Shap-E、Point-E），缺乏人体特有的层级结构建模能力，难以控制生成结果的可控性。
4. **SDS 梯度噪声导致几何模糊**：SDS 的随机性使得 3DGS 原有的密度控制机制不稳定，产生漂浮伪影和模糊的人体表面。

## 核心贡献（创新点）
1. **提出 Structure-Aware SDS 框架**：首次将 3DGS 有效适配到文本驱动 3D 人体生成任务，通过 SMPL-X 先验初始化高斯中心并结合 RGB-Depth 双分支 SDS 联合学习外观与几何。
2. **设计 Texture-Structure Joint Model**：扩展 Stable Diffusion 以同时去噪图像 RGB 与深度图，利用空间对齐的双模态分布引导高斯密度的自适应增长与剪枝，缓解稀疏梯度对几何优化的阻碍。
3. **发明 Annealed Negative Prompt Guidance**：将 SDS 分解为生成分数与分类器分数，仅利用更干净的分类器分数并以退火方式结合负提示，在低 CFG 尺度（7.5）下避免过饱和并获得真实外观。
4. **提出 Size-Conditioned Gaussian Pruning**：在密度控制后增设仅剪枝阶段，根据缩放因子阈值剔除低透明度大尺寸高斯，消除人体表面附近的漂浮伪影。

## 方法详解
### 1. 整体流程
基于 3D Gaussian Splatting，从文本提示生成高质量 3D 人体，包含高斯初始化、双分支 SDS 优化、退火负提示引导、尺寸条件剪枝四个核心模块。

### 2. 高斯初始化（SMPL-X Prior）
- 在 **SMPL-X** 网格表面均匀采样 **10 万** 个 3D 高斯中心，初始缩放为单位尺度，颜色取均值，不旋转，透明度设为 0.1。
- 相比通用先验（SfM、Shap-E），SMPL-X 提供更密集且结构连贯的人体拓扑，并覆盖手部与面部细节。
- 同时将 SMPL-X 关节映射为 COCO 风格关键点，剔除遮挡关节，作为姿态骨架图输入 SDS 模型。

### 3. Texture-Structure Joint Model（双分支扩散模型）
- 以 **SD 2.0** 为基础，复制 UNet 的 conv in、首个 DownBlock、最后一个 UpBlock 和 conv out 层，构建共享骨干网络。
- 同时去噪图像 x 和深度 d，输入条件为姿态图 p 和文本嵌入 y，采用 **v-prediction** 目标：

$$\mathbb{E}_{\mathbf{v}, t}\left[\|\mathbf{v}_{\phi}(\mathbf{x}_t; \mathbf{p}, y) - \mathbf{v}_t^{\mathbf{x}}\|_2^2 + \|\mathbf{v}_{\phi}(\mathbf{d}_t; \mathbf{p}, y) - \mathbf{v}_t^{\mathbf{d}}\|_2^2\right]$$

- 深度图由 MiDaS 在 LAION 数据集上标注生成。

### 4. Dual-Branch SDS（结构感知评分蒸馏采样）
同时优化 RGB 渲染和深度渲染，梯度公式为：

$$\nabla_\theta \mathcal{L}_{\text{SDS}} = \lambda_1 \cdot \mathbb{E}_{\epsilon_x, t}\left[w_t(\epsilon_\phi(\mathbf{x}_t; \mathbf{p}, y) - \epsilon_x)\frac{\partial \mathbf{x}}{\partial \theta}\right] + \lambda_2 \cdot \mathbb{E}_{\epsilon_d, t}\left[w_t(\epsilon_\phi(\mathbf{d}_t; \mathbf{p}, y) - \epsilon_d)\frac{\partial \mathbf{d}}{\partial \theta}\right]$$

- 其中 $\lambda_1 = \lambda_2 = 0.5$，深度渲染采用点基于 α-blending 的累积深度计算。
- 相比仅使用单模态（RGB 或 Normal/Depth）的方法，双分支提供空间对齐的联合分布指导。

### 5. Annealed Negative Prompt Guidance（退火负提示引导）
- 将标准 SDS 分解为生成分数 $\delta_g$ 和分类器分数 $\delta_c$，**仅使用更干净的分类器分数**以避免高方差噪声：

$$\delta_{nc} = \epsilon_\phi(\mathbf{x}_t; y) - \epsilon_\phi(\mathbf{x}_t; y_{\text{neg}})$$

- 引入退火策略：$\tau_2$ 在 timestep 200 之前从 1.0 线性降至 0.0，结合公式：

$$\nabla_\theta \mathcal{L}_{\text{SDS}} = \mathbb{E}_{\epsilon, t}\left[w_t(\tau_1 \cdot \delta_c - \tau_2 \cdot \delta_n)\frac{\partial \mathbf{x}}{\partial \theta}\right]$$

- CFG 尺度降至 **7.5**（DreamFusion 用 100），避免过饱和，同时负提示帮助排除不 desirable 属性。

### 6. Size-Conditioned Gaussian Pruning（尺寸条件剪枝）
- 在 2400–3300 次迭代期间（每 300 步执行一次），对缩放因子超过阈值 **0.008** 的高斯进行纯剪枝，剔除低透明度但尺寸过大的漂浮伪影。
- 得益于 SMPL-X 密集初始化，冗余高斯可在优化过程中补偿被剪除的部分。

## 实验与结果
### 数据集与实现
- 深度监督数据来自 **LAION** 数据集，由 **MiDaS** 标注深度图。
- 分辨率 512（SD 微调）/ 1024（3DGS 训练），Batch size=8，单张 **NVIDIA A100 (40GB)** 约 **1 小时** 完成优化。
- 3600 次迭代：300–2100 步进行密度控制（每 300 步 densify/prune），2400–3300 步进行 prune-only。

### 对比基线
- 通用文本转 3D：**DreamGaussian**、**GaussianDreamer**（均增强多视图/Point-E 先验）
- 专用人体生成：**TADA**、**DreamHuman**（结果取自原文）

### 定量结果（Table 1 & 2）
| 方法 | Text. Qual. | Geo. Qual. | Text Align. | CLIP↑ | Aes.↑ | HPSv2↑ |
|------|------------|-----------|-------------|-------|-------|--------|
| TADA | 3.76 | 3.53 | 4.35 | 30.13 | 5.709 | 0.253 |
| DreamHuman | 3.41 | 3.65 | 4.24 | 29.98 | 5.170 | 0.248 |
| DreamGaussian | 2.18 | 2.88 | 2.94 | 24.77 | 4.533 | 0.229 |
| GaussianDreamer | 2.94 | 3.00 | 3.06 | 25.12 | 4.641 | 0.231 |
| **HumanGaussian (Ours)** | **4.24** | **3.88** | **4.71** | **30.82** | **6.436** | **0.262** |

- 用户在纹理质量、几何质量、文本对齐三项全面领先，CLIP 得分 30.82 超越 TADA（30.13）。
- 美学得分（6.436）和 HPSv2（0.262）均为最高。

### 消融验证
逐层加入模块验证有效性：① 基础 3DGS → ② +SMPL-X 初始化+姿态条件 → ③ +退火负提示引导 → ④ +双分支 SDS → ⑤ +尺寸剪枝，每步均有显著提升。

## 相关工作脉络
1. **DreamFusion**：首次提出 SDS 将 2D 扩散先验蒸馏到 3D 表示，本文在其基础上改进梯度稳定性和过饱和问题。
2. **3D Gaussian Splatting (3DGS)**：高效显式 3D 表示，本文首次将其系统性地引入文本驱动 3D 人体生成领域。
3. **DreamHuman**：基于 imGHUM+NeRF 的文本驱动人体生成，效率低且难以建模细粒度细节；本文用 3DGS 替代并实现更快训练。
4. **TADA**：基于 SMPL-X 变形+UV 纹理优化的两阶段方法，依赖 DMTet 表示，无法高效捕捉细粒度细节；本文端到端优化 3DGS。
5. **HumanNorm**：同期工作，微调 text-to-normal/depth 两个独立模型做 SDS，存在深度与法向不对齐问题；本文使用空间对齐的双分支联合模型。
6. **GaussianDreamer / DreamGaussian**：通用文本转 3DGS 方法，本文将其扩展至人体域并加入结构化先验与负提示引导。

## 局限性与未来方向
1. **手部和脚部生成质量有限**：受限于现有 2D 文本到图像模型在手/脚区域的表现，有时无法忠实渲染这些部位。
2. **依赖 SMPL-X 先验**：初始化锚定在参数化人体模型上，对于非标准人体形态或极端姿势可能泛化受限。
3. **深度监督依赖 MiDaS 预标注**：LAION 上的深度标签质量影响双分支 SDS 的效果上限。
4. **未来方向**：作者计划探索改进手/脚生成质量，以及扩展到其他人体相关应用。

## 研究启发与可借鉴点
1. **结构先验初始化策略可迁移**：SMPL-X 密集采样初始化高斯中心的方法，可推广到其他具有参数化模型的对象生成（如动物、角色）。
2. **双分支 SDS 设计思路**：同时蒸馏 RGB 与 Depth 的联合分布，可用于任意需要几何+外观联合优化的 3D 生成任务。
3. **退火负提示引导的技术路径**：将 SDS 分解为生成分数与分类器分数，仅使用更稳定的分类器分数并结合退火负提示，可直接复用到其他文本到 3D 生成框架。
4. **尺寸条件剪枝机制**：基于高斯缩放因子阈值的伪影消除策略，是一种轻量且有效的后处理技巧，适用于各类 3DGS 优化任务。
5. **姿态条件注入方式**：将骨架关键点图通过通道拼接输入扩散模型，为条件生成提供了简洁有效的结构控制方案。

## 关键术语表
**3D Gaussian Splatting (3DGS)**：一种基于各向异性高斯椭球的显式 3D 场景表示，支持实时渲染和快速收敛，通过 tile-based rasterization 实现高效合成。
**Score Distillation Sampling (SDS)**：DreamFusion 提出的核心技术，通过蒸馏 2D 预训练扩散模型的分数信息来优化 3D 场景表示。
**Classifier-Free Guidance (CFG)**：扩散模型中通过无条件和有条件预测的差值来增强生成质量的引导技术，尺度 τ 控制引导强度。
**SMPL-X**：扩展版 SMPL 参数化人体模型，包含 10,475 个顶点和 54 个关键点，覆盖身体、手部和面部的形状拓扑。
**v-prediction**：渐进蒸馏训练目标，将噪声预测转化为 v = α_t · ε - σ_t · x 形式，提升扩散模型训练稳定性。
**Annealed Negative Prompt Guidance**：本文提出的负提示退火策略，在较早 timestep 使用负提示抑制不 desirable 特征，随优化推进逐渐降低其权重。

## 可复现要素
- **数据集**：LAION-5B（用于训练双分支扩散模型，深度由 MiDaS 标注）；人体生成评估使用 DreamHuman 数据集的 30 个提示。
- **代码开源**：项目页面 https://alvinliu0.github.io/projects/HumanGaussian；基于 **PyTorch** + **ThreeStudio** 框架实现。
- **权重开源**：论文未明确说明是否开源预训练权重。
- **关键超参**：
  - 高斯初始化数量：100k
  - 初始透明度：0.1
  - SH 阶数：degree 0
  - 训练迭代：3600 步
  - Densify/Prune 区间：300–2100 步（间隔 300）
  - Prune-only 区间：2400–3300 步
  - 缩放阈值：0.008
  - λ₁ = λ₂ = 0.5
  - τ₁ = 7.5, τ₂ 从 1.0 退火至 0.0（timestep 200）
  - 训练分辨率：1024，Batch size=8
  - 学习率：μ=5e-5, s=1e-3, q=1e-2, c=1.25e-2, α=1e-2
