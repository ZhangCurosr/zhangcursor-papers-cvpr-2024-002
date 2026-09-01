---
title: "ReGenNet-Towards-Human-Action-Reaction-Synthesis"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Xu_ReGenNet_Towards_Human_Action-Reaction_Synthesis_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 15:21:46"
---

# 论文速读：ReGenNet-Towards-Human-Action-Reaction-Synthesis

## 一句话总结
本文针对人类因果交互中发起者（actor）与反应者（reactor）的角色不对称性，提出首个带actor-reactor标注的多人动作-反应合成基准与ReGenNet模型；该模型基于扩散过程与带掩码的Transformer解码器，在actor未来动作完全不可见的在线设定下，实时生成语义合理、空间约束一致的人体反应序列。

## 研究问题与动机
- **角色不对称性被忽视**：现有以人为中心的生成模型多聚焦人与静态场景/物体交互，极少建模双人因果链中actor主动、reactor被动响应的非对称关系。
- **数据集缺乏方向标注**：主流多人交互数据集（NTU120、InterHuman、Chi3D等）仅记录耦合动作对，未区分动作发起与响应顺序，导致无法训练条件反应生成器。
- **在线同步生成难度大**：真实交互要求reactor即时响应（online），模型不能预知actor未来帧；多数现有方法为离线批处理或仅处理单主体运动，无法满足低延迟同步需求。
- **细粒度与物理合理性不足**：稀疏骨架或基础SMPL模型难以表达手部精细交互与面部表情；纯数据驱动的扩散生成易出现空间穿模或相对位姿违背直觉的问题。

## 核心贡献（创新点）
- 提出首个含actor-reactor角色标注的多人动作-反应合成基准：对NTU120、InterHuman、Chi3D进行不对称性标注与SMPL-X扩展，填补现有数据集在因果角色方向上的空白。
- 设计ReGenNet扩散生成框架：采用Transformer解码器结构配合左向方向性掩码，天然契合因果推理，区别于MDM等自回归/双向架构，杜绝未来actor信息泄露。
- 引入显式基于距离的交互损失（$\mathcal{L}_{inter}$）：通过正向运动学与旋转矩阵直接度量双方在pose、orientation、translation上的相对时空偏差，将几何先验硬约束融入生成目标，避免纯数据驱动的穿模与空间违背。
- 构建模块化多设置统一架构：默认移除意图分支实现无约束（unconstrained）在线生成；可通过关闭掩码切换至离线（offline）模式，或加入意图标签切换至受限（constrained）模式，一套权重兼容多种应用场景。

## 方法详解
- **数据表征**：采用SMPL-X参数化人体序列 $x^{1:N}$，每帧 $x^i = [\theta_i^x, q_i^x, \gamma_i^x]$（姿态 $\theta \in \mathbb{R}^{3K}$、全局朝向 $q \in \mathbb{R}^3$、根平移 $\gamma \in \mathbb{R}^3$，K=54含手指与面部关节）。
- **扩散生成流程**：前向过程按Eq.(1)逐步加高斯噪声；反向去噪网络 $F$ 直接预测清洁姿态 $\hat{x}_0 = F(x_t, y^{1:N}, t, a)$，训练目标为 $\mathcal{L}_{dm} = \mathbb{E}_{x_0,t}[\|x_0 - F(x_t, y, t, a)\|_2^2]$。
- **条件Token构建**：将noised反应 $x_t^{1:N}$ 与动作条件 $y^{1:N}$ 分别经FC投影至维度 $d$ 后拼接，再经FC融合为 $z_t^{1:N}$；时间步 $t$ 与可选意图 $a$ 经独立FFN投影后相加得 $z_{at}$。
- **Transformer Decoder与在线掩码**：堆叠 $\ell_{dec}=8$ 层Decoder Unit，利用方向性掩码（mask）强制多头注意力仅关注历史与当前token；推理时采用滑动窗口自回归策略，实现逐帧即时生成。
- **显式交互损失**：计算真实反应 $x_0$ 与预测反应 $\hat{x}_0$ 相对于actor $y$
