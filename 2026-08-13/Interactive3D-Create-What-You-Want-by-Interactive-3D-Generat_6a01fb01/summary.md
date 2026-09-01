---
title: "Interactive3D-Create-What-You-Want-by-Interactive-3D-Generat"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Dong_Interactive3D_Create_What_You_Want_by_Interactive_3D_Generation_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 20:14:21"
field: "3D生成与编辑"
keywords: ["3D生成", "交互式3D", "Gaussian Splatting", "InstantNGP", "SDS Loss", "3D编辑"]
innovations: ["双阶段交互3D生成框架（Gaussian Splatting交互+InstantNGP细化）", "Interactive Hash Refinement模块解决局部细化哈希冲突", "将DragGAN拖拽思想推广至3D空间的形变/刚性拖拽操作"]
benchmarks: ["CLIP R-Precision on Cap3D-derived prompts"]
---

# 论文速读：Interactive3D: Create What You Want by Interactive 3D Generation

## 一句话总结
Interactive3D 提出了一种两阶段交互式3D生成框架：第一阶段利用 Gaussian Splatting 允许用户直接对3D对象进行添加/移除部件、形变拖拽、语义编辑等操作；第二阶段将 Gaussian 蒸馏为 InstantNGP 并通过 Interactive Hash Refinement 模块进一步细化细节与提取高质量网格。

## 研究问题与动机
1. **现有3D生成方法缺乏精确控制**：依赖初始文本提示或2D参考图像的方法控制力有限，无法支持用户在生成中途进行灵活修改。
2. **3D控制信号收集困难**：直接适配 ControlNet 的思路受限于3D条件数据集的稀缺，以及缺乏如 Stable Diffusion 级别的3D基础模型。
3. **表征互补性未被充分利用**：Gaussian Splatting 便于直接交互编辑，但难以提取高质量网格；InstantNGP 擅长几何细化，但作为隐式表示难以直接交互。

## 核心贡献（创新点）
1. **交互式3D生成框架设计**：首次将多种3D交互操作（添加/移除、拖拽、语义编辑）直接集成到生成过程中，与仅依赖初始文本/参考图的方法本质不同。
2. **Gaussian Splatting + InstantNGP 双阶段协同策略**：利用 Gaussian 的显式特性支持交互，再通过 NeRF 蒸馏转换为 InstantNGP 用于几何精化和网格提取，解决了单一表征的局限。
3. **Interactive Hash Refinement 模块**：通过部分特定的多级哈希表映射解决局部细化时的哈希冲突问题，使模型可聚焦于用户指定区域而避免其他部分退化，区别于全局 Fine-tune 方法。

## 方法详解
**Stage I — 基于 Gaussian Splatting 的交互生成：**
- **部件添加/移除**：将 Gaussian 集合拼接实现添加；通过2D渲染图点选+SAM分割器投影多视角确定部件索引，或直接3D点选删除。
- **几何变换**：对选定部件构建包围盒，应用旋转/平移/拉伸等操作后与原集合拼接。
- **可变形与刚性拖拽**：灵感来自 DragGAN，选取源点 $p_s$、目标点 $p_t$ 和半径 $r$，计算激活部件 $P=\{e_i|\|\mu_i-p_s\|_2\leq r\}$，每步施加偏移 $\mu_i'=\mu_i+\alpha\frac{(p_t-p_s)}{\|p_t-p_s\|_2}$，引入运动监督损失 $L_{\text{motion}}=\sum_{i\in I(P)}\|\mu_i'-p_t\|_1$；刚性拖拽额外添加约束损失 $L_{\text{rigid}}=\sum_{i\in I(P)}|\|p_s-p_i\|_2-\|p_s^*-p_i^*\|_2|$。
- **语义编辑**：选择部件后输入新文本提示，仅对该区域计算 SDS 损失。
- **交互式 SDS Loss**：采用自适应相机放大策略（将相机置于修改区域附近进行局部渲染），并允许用户调整去噪步长 $t$ 以适配生成阶段。

**Stage II — 基于 InstantNGP 的细化：**
- **NeRF 蒸馏**：用 Gaussian 渲染图像监督 InstantNGP 的随机视角渲染，损失为 $L_{\text{distill}}=\frac{1}{M}\sum_{c\sim C(\theta,\phi)}\|\mathcal{V}(\mathcal{F},c)-\mathcal{R}(\mathcal{E},c)\|_1$。
- **Interactive Hash Refinement**：从转换后的场中提取 $32\times32\times32$ 的二值占用网格 $O$，对用户选定区域 $Q$ 取交集 $O^{\text{part}}=O\cap Q$，构建部分特定的多级哈希表 ${\mathcal{H}}^{\text{part}}$ 和特征 ${\mathcal{F}}^{\text{part}}$，引入轻量 MLP 预测残差颜色和密度，避免全局哈希冲突。

## 实验与结果
- **数据集**：使用 Cap3D 派生的 50 个文本提示词进行评估。
- **评估指标**：CLIP R-Precision。
- **主要结果**（Table 1）：

| 方法 | R-Precision | 平均耗时 |
|------|------------|---------|
| DreamFusion | 0.67 | 1.1h |
| ProlificDreamer | 0.83 | 3.4h |
| **Ours** | **0.94** | **50min** |

- Interactive3D 以 0.94 的 CLIP R-Precision 超越所有基线，同时将总训练步数控制在 20k（Stage I 和 Stage II 各 10k），总耗时约 50 分钟，显著快于基线方法。

## 相关工作脉络
1. **DreamFusion [27]**：基于 SDS loss 的文本到3D优化方法，仅依赖初始文本提示，无中途交互能力。
2. **ProlificDreamer [35]**：改进 SDS loss 提升3D生成质量，但仍局限于文本/参考图驱动，无法支持局部修改。
3. **DragGAN [26]**：2D图像上的点驱动形变编辑，本文将其思想推广至3D空间并设计了对应的运动监督与刚性约束损失。
4. **Gaussian Splatting [11]**：显式3D高斯表示，本文利用其可编辑性实现交互，区别于其在三维重建中的传统用法。
5. **InstantNGP [23]**：多级哈希编码的 NeRF 加速方法，本文将其用于几何细化阶段，并针对性地解决其交互缺陷。
6. **PointE / Shap-E [10, 25]**：前馈式3D生成方法，速度快但控制粒度粗，本文方法适合需要精细控制的高质量生成场景。

## 局限性与未来方向
1. 依赖2D扩散模型（SDS loss）作为先验，3D一致性仍受限于2D prior 的质量。
2. 交互操作可能引入非预期伪影，需要后续 refinement 步骤补偿。
3. 当前框架主要针对单对象生成，未涉及复杂场景或多对象交互。
4. 部件选择依赖 SAM 等外部分割器，在早期生成阶段（渲染信息不足时）精度受限。

## 研究启发与可借鉴点
1. **双表征协同范式**：Gaussian Splatting（交互友好）+ InstantNGP（几何友好）的组合策略可迁移至其他需要交互+高质输出的3D生成任务。
2. **部分特定的哈希映射设计**：Interactive Hash Refinement 解决局部细化时"牵一发而动全身"的问题，思路可扩展到其他哈希编码的隐式表示系统中。
3. **交互式 SDS Loss 设计**：自适应相机缩放+可调去噪步长 $t$ 的机制，可作为通用工具集成到任何基于 SDS 的3D优化流程中。
4. **DragGAN to 3D 的迁移**：刚性/可变形拖拽的损失设计（运动监督+刚性约束）可直接借鉴到3D点云/网格编辑任务中。

## 关键术语表
**Gaussian Splatting**：用一组3D高斯椭球体显式表示场景并通过溅射渲染的3D表示方法，适合直接编辑。
**SDS Loss (Score Distillation Sampling)**：利用预训练的2D扩散模型梯度来指导3D表示优化的损失函数。
**InstantNGP**：基于多级哈希表编码的快速 NeRF 实现，支持高效渲染和网格提取。
**Interactive Hash Refinement**：本文提出的局部细化模块，通过部分特定的多级哈希表为选中区域添加残差特征。
**Deformable/Rigid Dragging**：分别对应局部平滑形变和整体刚体移动的3D拖拽交互操作。
**Part-specific Hash Mapping**：为特定区域构建独立哈希表，避免与其他区域共享特征导致的冲突退化。
**CLIP R-Precision**：评估生成结果与文本提示对齐程度的指标，衡量检索准确性。

## 可复现要素
- **数据集**：Cap3D（论文未提及是否公开，但其派生的50个提示词已列出）
- **代码/权重**：项目网页 https://interactive-3d.github.io/，论文未明确说明代码开源状态
- **关键超参**：每对象总训练步数 20k（Stage I 10k + Stage II 10k）；Occupancy grid 分辨率 $32\times32\times32$；默认使用 2 个视图进行部件选取
- **硬件**：NVIDIA A100 GPU
- **实现框架**：Stage I 基于 [4]（Text-to-3D using Gaussian Splatting），Stage II 基于 [23]（InstantNGP）
