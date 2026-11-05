---
title: "SDPose-Tokenized-Pose-Estimation-via-Circulation-Guide-Self"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_SDPose_Tokenized_Pose_Estimation_via_Circulation-Guide_Self-Distillation_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 16:37:49"
field: "轻量级姿态估计"
keywords: ["Human Pose Estimation", "Self-Distillation", "Transformer", "Lightweight Model", "Edge Deployment"]
innovations: ["首创循环前向提升Transformer隐式深度", "同向量子空间自蒸馏无需额外对齐", "同等规模模型SOTA且可迁移至分类任务"]
benchmarks: ["MSCOCO validation", "MSCOCO test-dev", "Crowdpose test"]
---

# 论文速读：SDPose-Tokenized-Pose-Estimation-via-Circulation-Guide-Self

## 一句话总结
论文提出SDPose方法，通过Multi-Cycled Transformer（MCT）模块实现循环前向传递，结合自蒸馏范式将知识从多轮推理中提取到单次前向模型中，在不增加参数量和计算量的前提下显著提升小尺度Transformer姿态估计模型的性能。

## 研究问题与动机
1. **边缘部署困境**：当前SOTA的Transformer姿态估计模型（如ViTPose）参数量达6.32亿、计算量达122.9 GFLOPs，无法部署在边缘计算设备上。
2. **小模型欠拟合**：当Transformer backbone规模不足时，模型难以有效学习关键点与视觉元素间的关系，导致性能大幅下降。
3. **传统蒸馏方法的局限**：现有蒸馏方法需要额外特征空间对齐操作（可能降低性能），或需要训练庞大的教师网络（额外训练成本）。
4. **计算效率与性能平衡**：需要一种方法在保持轻量级的同时，突破小模型的性能瓶颈。

## 核心贡献（创新点）
1. **首创循环前向提升隐式深度**：首次发现让token在Transformer层间循环传递可在不增加参数的情况下提升模型隐式深度（latent depth），并据此设计了MCT模块。
2. **自蒸馏范式设计**：设计了SDPose自蒸馏框架，利用同向量子空间中前后cycle输出的天然对齐性，无需额外操作即可将多轮推理知识蒸馏到单次前向模型中。
3. **轻量级SOTA突破**：在MSCOCO验证集上，SDPose-S-V2以6.2M参数、4.7 GFLOPs达到73.5% mAP，成为同等规模模型的新SOTA；SDPose-T仅用4.4M参数、1.8 GFLOPs即达到69.7% mAP。
4. **跨任务泛化验证**：将方法迁移至图像分类任务（DeiT-Tiny），证明MCT和自蒸馏策略的通用性。

## 方法详解
**MCT模块设计**：
- 基于TokenPose框架，将视觉tokens（VT）和可学习关键点tokens（KT）拼接后输入Transformer编码器
- 设计N次循环前向机制：第i次循环的输入为$(VT_{i-1}, KT_{i-1})$，输出为$(VT_i, KT_i)$
- 最终使用$VT_N, KT_N$进行预测，实现"等效更深Transformer"的效果

**自蒸馏机制**：
- 训练阶段：所有cycle的输出都参与蒸馏，用后cycle结果指导前cycle学习
- 推理阶段：仅执行单次前向，无额外计算开销
- 可视化证明：第一cycle注意力集中在局部区域，第二cycle注意力扩展到全局，说明每轮携带丰富信息

**损失函数**：
- Keypoint token蒸馏损失：$L_{kt} = \sum_{i=1}^{N-1} MSE(KT_i, KT_{i+1})$
- Visual token蒸馏损失：$L_{vt} = \sum_{i=1}^{N-1} MSE(VT_i, VT_{i+1})$
- 预测损失（各cycle预测头均约束）：$L_{pose} = \sum_{i=1}^{N} MSE(P_i, GT)$
- 总损失：$L = L_{pose} + \alpha_1 L_{kt} + \alpha_2 L_{vt}$，其中$\alpha_1 = \alpha_2 = 5e-6$

## 实验与结果
**数据集**：MSCOCO（train2017/val2017/dev2017）、Crowdpose

**主要结果（MSCOCO验证集，heatmap-based）**：
| 模型 | 参数量 | GFLOPs | AP | 提升 |
|------|--------|--------|-----|------|
| TokenPose-S-V1 | 6.6M | 2.4 | 69.5% | - |
| **SDPose-T** | **4.4M (↓33.3%)** | **1.8 (↓25.0%)** | **69.7% (↑0.2%)** | - |
| TokenPose-S-V2 | 6.2M | 4.7 | 71.8% | - |
| **SDPose-S-V2** | **6.2M** | **4.7** | **73.5% (↑1.7%)** | 同规模SOTA |
| TokenPose-B | 13.2M | 5.2 | 73.2% | - |
| **SDPose-B** | **13.2M** | **5.2** | **73.7% (↑0.5%)** | - |

**Crowdpose测试结果**：SDPose-S-V2达到64.5% AP（↑2.2%），验证泛化性。

**消融实验关键发现**：
- 不使用蒸馏、仅用第一cycle预测：AP仅55.4%（严重下降）
- 完整三损失组合：达到最优72.3%
- 循环次数超过2次时性能下降（过度增强全局信息可能丢失局部细节）
- 与PPT结合：计算量降至PPT-S水平，性能保持SDPose-S-V1水平

## 相关工作脉络
1. **TokenPose [3]**：提出关键点token表示的heatmap-based方法，本文在此基础上改进为MCT版本
2. **DistilPose [8]**：首次将知识蒸馏应用于HPE，但需单独训练教师网络；本文用自蒸馏避免此开销
3. **PPT [14]**：通过丢弃低注意力token减少计算，但性能有所下降；本文可与PPT结合获得更优效果
4. **OKDHP [7]**：在线知识蒸馏方法，参数量大（13-24M）、计算量高（25-47 GFLOPs）
5. **ViTPose [5]**：大规模SOTA方法（632M参数），无法部署于边缘设备
6. **Be Your Own Teacher [16]**：CNN自蒸馏工作，本文首次将其扩展到Transformer-based HPE

## 局限性与未来方向
1. **循环次数限制**：实验表明循环次数>2时性能下降，过度增强全局信息可能丢失关键点局部细节，需要更精细的cycle控制策略
2. **训练稳定性**：自蒸馏需要多cycle同时优化，对超参数（如$\alpha_1, \alpha_2$）较敏感
3. **推理延迟未充分讨论**：虽然FLOPs不变，但循环前向在训练时的实现方式未详细说明
4. **3D姿态估计扩展性**：仅在2D任务验证，3D HPE的token表示不同，泛化能力待验证
5. **极端小样本场景**：方法在极小模型（<4M参数）上的表现未充分探索

## 研究启发与可借鉴点
1. **隐式深度提升策略**：循环前向传递的思想可迁移至其他视觉任务（检测、分割），通过复用参数实现"深度增强"而非"宽度扩展"
2. **自蒸馏的向量空间对齐优势**：利用Transformer同层输入输出在同一向量空间的特性，避免额外对齐操作，这一设计简洁且高效
3. **多尺度监督信号**：每个cycle都通过预测头计算损失的策略，可借鉴于其他多阶段推理框架，提供渐进式监督
4. **参数利用率可视化分析**：通过可视化近零参数比例验证训练充分性，这一分析方法可复用于其他模型效率研究
5. **与剪枝方法的兼容性**：证明MCT可与PPT等剪枝策略结合，为后续"增深+减宽"的协同优化提供思路

## 关键术语表
**SDPose**：本文提出的自蒸馏姿态估计框架，通过循环前向+知识蒸馏提升小模型性能
**MCT (Multi-Cycled Transformer)**：多循环Transformer模块，token在相同层间多次循环传递
**Latent Depth（隐式深度）**：完整推理过程中实际参与的Transformer层深度，本文通过循环而非加深网络提升该值
**Tokenized Pose Estimation**：将关键点表示为可学习token，与视觉token交互进行姿态估计的范式
**Self-Distillation（自蒸馏）**：同一模型内部不同阶段/层级之间的知识蒸馏，无需外部教师网络
**Keypoint Token (KT)**：表示人体关键点的可学习向量，与视觉token共同输入Transformer
**Visual Token (VT)**：图像patches经过线性投影后的视觉特征token

## 可复现要素
- **数据集**：MSCOCO（公开）、Crowdpose（公开）
- **代码**：论文未明确开源声明，实验基于MMPose框架[19]
- **权重**：论文未提供预训练权重下载链接
- **关键超参**：
  - 训练轮数：300 epochs
  - 优化器：Adam，初始学习率1e-3，epoch 200和260各衰减10倍
  - 损失权重：$\alpha_1 = \alpha_2 = 5e-6$
  - Batch size：64 samples/GPU
  - 硬件：8× NVIDIA Tesla V100
  - 输入尺寸：256×192
- **基线复现**：文中部分基线（TokenPose、PPT等）使用*标记并在MMPose上重新训练评估
