---
title: "SDPose-Tokenized-Pose-Estimation-via-Circulation-Guide-Self"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_SDPose_Tokenized_Pose_Estimation_via_Circulation-Guide_Self-Distillation_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 15:22:54"
field: "轻量化视觉模型"
keywords: ["人体姿态估计", "Transformer", "自蒸馏", "模型轻量化", "边缘计算"]
innovations: ["通过循环前向增加transformer潜在深度无需额外参数", "循环引导的自蒸馏范式提取多轮推理知识到单次前向"]
benchmarks: ["MSCOCO", "Crowdpose"]
---

# 论文速读：SDPose-Tokenized-Pose-Estimation-via-Circulation-Guide-Self

## 一句话总结
本文提出SDPose，一种通过循环引导自蒸馏提升小型transformer人体姿态估计(HPE)模型性能的方法。核心设计包括Multi-Cycled Transformer (MCT)模块，通过多次循环前向增加模型潜在深度而无需额外参数，并配合自蒸馏将多轮知识提取到单次前向，实现参数量与性能的均衡。

## 研究问题与动机
- **小型transformer模型的欠拟合困境**：现有SOTA transformer-based HPE模型参数庞大（如ViTPOSE约632M参数、122.9B FLOPs），难以部署到边缘计算设备；缩小规模后由于能力不足，难以有效学习关键点与视觉元素间的复杂关系，导致性能显著下降。
- **现有蒸馏方法的局限**：传统教师-学生蒸馏需要额外训练大规模教师网络，成本高昂；特征蒸馏需对齐向量空间，操作复杂且可能导致性能下降。
- **参数利用不充分**：可视化显示TokenPose等小型transformer模型各层存在较多接近零的参数，表明模型训练不充分，潜力未完全释放。

## 核心贡献（创新点）
- **首次发现循环前向可提升潜在深度**：提出Multi-Cycled Transformer (MCT)模块，通过让token在transformer层中多次循环，在不增加参数的前提下等价于更深层的transformer网络，突破小型模型欠拟合限制。与已有工作的本质区别在于，之前工作仅通过堆叠层数增加深度（同步增加参数），而本文在不增加参数的情况下实现深度等效。
- **循环引导的自蒸馏范式**：设计自蒸馏方案，利用MCT中每循环输入输出在同一向量空间的特点，直接用后一循环输出蒸馏前一循环，无需额外对齐操作即可提取完整推理知识到单次前向模型。这是首次将自蒸馏应用于transformer-based HPE任务。
- **轻量高效的SOTA性能**：在相同参数量和计算量约束下，相比基线模型取得显著提升（SDPose-S-V2在6.2M参数、4.7 GFLOPs下达到73.5% mAP，较TokenPose-S-V2提升1.7%），在同规模小模型中建立新的SOTA。

## 方法详解
- **Multi-Cycled Transformer (MCT) 模块**：遵循TokenPose框架，将图像特征划分为visual tokens (VT) 并拼接K个可学习keypoint tokens (KT)，送入transformer encoder。MCT将tokens通过N次循环：第i次循环以$VT_{i-1}, KT_{i-1}$为输入，输出$VT_i, KT_i$，最终使用最后一次循环的输出进行预测。多次循环使token获得更全局的信息交互。
- **自蒸馏机制**：训练时，利用第i+1循环的输出蒸馏第i循环的输出，因为同一transformer层内输入输出在同一向量空间，无需额外变换即可直接对比。具体地，对keypoint tokens和visual tokens分别计算MSE损失：
  - $L_{kt} = \sum_{i=1}^{N-1} MSE(KT_i, KT_{i+1})$
  - $L_{vt} = \sum_{i=1}^{N-1} MSE(VT_i, VT_{i+1})$
- **预测损失**：所有循环的输出都通过预测头生成预测结果$P_i$，并与ground truth计算损失：
  - $L_{pose} = \sum_{i=1}^{N} MSE(P_i, GT)$
- **总损失函数**：$L = L_{pose} + \alpha_1 L_{kt} + \alpha_2 L_{vt}$，其中$\alpha_1 = \alpha_2 = 5e-6$。推理时仅执行单次前向传播，不增加额外计算开销。

## 实验与结果
- **数据集**：MSCOCO（57K训练图，val2017和dev2017评估）和Crowdpose（20K crowded场景图，约80K人，14个关键点）。
- **主要结果（MSCOCO val，输入尺寸256×192）**：
  - **SDPose-T**：4.4M参数（↓33.3%）、1.8 GFLOPs（↓25.0%），AP=69.7%（↑0.2% vs TokenPose-S-V1）。
  - **SDPose-S-V1**：6.6M参数、2.4 GFLOPs，AP=72.3%（↑2.8% vs TokenPose-S-V1）。
  - **SDPose-S-V2**：6.2M参数、4.7 GFLOPs，AP=73.5%（↑1.7% vs TokenPose-S-V2），**同规模小模型中SOTA**。
  - **SDPose-B**：13.2M参数、5.2 GFLOPs，AP=73.7%（↑0.5% vs TokenPose-B）。
- **回归方法对比（MSCOCO test-dev）**：SDPose-Reg（5.4M参数、2.4 GFLOPs）达到72.1% AP，较DistilPose-S提升1.1%。
- **Crowdpose结果**：SDPose-S-V1和SDPose-S-V2分别较基线提升1.6%和2.2% AP。
- **消融实验**：
  - 三种损失联合使用效果最佳（Table 5）。
  - 循环次数：2次循环优于3次循环（过深可能导致局部信息遗忘）。
  - 扩展性：与PPT结合可在保持SDPose-S-V1性能的同时将FLOPs降至PPT-S水平。

## 相关工作脉络
- **TokenPose [3]**：本文基础模型，采用可学习keypoint tokens与transformer编码器进行姿态估计，是SDPose的主要对比基线。SDPose在其基础上引入MCT和自蒸馏实现性能提升。
- **PPT [14]**：通过丢弃低注意力图像token降低计算复杂度，但牺牲性能。本文方法与PPT可结合（Table 8），证明兼容性。
- **DistilPose [8]**：将heatmap-based教师知识蒸馏到regression-based学生，需要单独训练教师网络。本文的自蒸馏无需额外教师，更为轻量。
- **ViTPOSE [5]**：纯transformer架构，参数量632M、122.9B FLOPs，代表高性能但重计算的transformer HPE方法。本文聚焦小型化场景。
- **OKDHP [7]**：在线知识蒸馏姿态估计，需要额外教师网络。本文的自蒸馏消除了这一负担。

## 局限性与未来方向
- **循环次数需调优**：消融显示3次循环反而不如2次循环，过多循环可能导致关键局部信息丢失，如何自适应选择循环次数值得研究。
- **仅验证了特定模型结构**：主要基于TokenPose和DistilPose变体，对其他先进小型transformer架构的通用性有待进一步验证。
- **自蒸馏仅应用于HPE**：虽然扩展到分类任务（Deit-Tiny）有正收益，但在更广泛的视觉任务上的潜力尚未充分探索。

## 研究启发与可借鉴点
- **"潜在深度"概念**：通过循环前向而非增加层数来提升模型有效深度，这一思路可迁移到其他token-based视觉任务（如目标检测、分割），用于提升小型模型性能。
- **向量空间一致的自蒸馏**：利用transformer层输入输出同空间的特性直接蒸馏，避免了特征对齐的额外开销，这一设计可推广到任意基于transformer的架构。
- **参数分布可视化分析**：通过可视化transformer层参数分布（接近零的参数比例）来量化训练充分程度，为模型诊断提供了直观有效的工具。
- **与剪枝/稀疏化方法的兼容性**：本文与PPT的结合证明了SDPose可与其它轻量化工具联合使用，为构建极致高效的HPE管线提供了思路。

## 关键术语表
- **潜在深度 (Latent Depth)**：定义为由完整推理过程中实际参与计算的transformer层深度，通过循环前向可在不增加参数的前提下提升潜在深度。
- **Multi-Cycled Transformer (MCT)**：核心模块，让tokenized特征多次循环通过transformer层，每次循环输出作为下一次输入。
- **自蒸馏 (Self-Distillation)**：在同一模型内部，用后一循环的输出作为教师蒸馏前一循环的知识，无需额外教师网络。
- **Keypoint Tokens (KT)**：可学习的token，用于表示各个身体关键点信息。
- **Visual Tokens (VT)**：由图像patch经线性投影生成的特征token。
- **TokenPose**：本文的基础模型，采用keypoint tokens与transformer编码器进行姿态估计的轻量化方法。

## 可复现要素
- **数据集**：MSCOCO（公开）、Crowdpose（公开）
- **代码**：论文未明确声明开源，但实验基于MMPose框架
- **关键超参**：循环次数N=2（最优），训练epoch=300，初始学习率1e-3，loss权重α₁=α₂=5e-6，优化器Adam，8× NVIDIA Tesla V100 GPU，batch size=64 per GPU
