---
title: "Multi-scale-Dynamic-and-Hierarchical-Relationship-Modeling-f"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_Multi-scale_Dynamic_and_Hierarchical_Relationship_Modeling_for_Facial_Action_Units_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 12:28:23"
field: "面部动作单元识别"
keywords: ["Action Unit Recognition", "Facial Dynamics", "Multi-scale Modeling", "Hierarchical Relationship", "Spatio-temporal Learning", "Graph Attention"]
innovations: ["多尺度时序差分自适应加权模块（MFD），首次按AU异质性分别建模不同空间尺度的面部动态", "两阶段分层时空关系建模（HSR），先局部后跨区域显式学习AU层级依赖", "在BP4D和DISFA上同时达到SOTA，MFD与HSR模块互补贡献验证充分"]
benchmarks: ["BP4D", "DISFA"]
---

# 论文速读：Multi-scale-Dynamic-and-Hierarchical-Relationship-Modeling-f

## 一句话总结
本文提出了 **MDHR**（Multi-scale Dynamic and Hierarchical Relationship）模型，通过多尺度时序差分模块（MFD）自适应捕捉不同空间尺度下的面部动态，并以两阶段策略分层建模局部/跨区域的时空AU关系，在 BP4D 和 DISFA 数据集上均达到新的 SOTA。

## 研究问题与动机
1. **层级关系建模缺失**：现有方法（Transformer、Graph-based）对AU间关系采用统一策略，未显式考虑同一/邻近面部区域的AU关联更强这一自然层级结构（Problem 1）。
2. **多尺度动态异质性忽略**：不同AU激活范围与幅度差异显著（如AU25为大尺度口部变形，AU2为眉毛细微运动），现有时序建模方法无法自适应区分各空间尺度的贡献（Problem 2）。
3. **面部动态捕捉不精细**：传统3D CNN/LSTM对细微肌肉运动敏感度不足；光流等方法未考虑不同AU对应动态在范围与幅度上的异质性。
4. **静态方法局限**：纯静态图像方法忽略时序线索；基于Patch的方法受限于地标检测误差且忽略上下文。

## 核心贡献（创新点）
1. **提出 MFD 模块**：首个针对每个AU在每个空间尺度上自适应强调相关面部动态的模块，利用自适应加权块解决不同AU激活范围/幅度异质性问题。
2. **提出 HSR 模块**：首个以两阶段方式分层学习局部与跨区域时空关系的模块，先建模同区域AU关联，再通过图注意力网络显式学习跨区域AU依赖。
3. **达到新 SOTA**：在 BP4D（66.6% F1，ResNet-50）和 DISFA（66.2% F1，Swin-B）上均超越所有现有静态/时空方法，MFD 和 HSR 模块互补贡献验证充分。

## 方法详解

### 整体流程
输入连续 $T$ 帧面部视频，以中心帧 $f^t$ 及其前后 $k$ 帧为窗口，经 Backbone 提取 $L$ 层多尺度静态特征 $X_l$，依次送入 **MFD** 和 **HSR** 模块，最后由 TCN + 相似度计算（SC）预测每帧的 N 个 AU。

### MFD 模块（Multi-scale Facial Dynamic Modelling）
- **多尺度时序差分（TD）层**：对相邻帧的 $l$ 层特征图逐点相减 $d_l^t = x_l^t - x_l^{t-1}$，捕获该尺度的面部动态；经 Conv2D（kernel=8/l）统一形状后，沿时间轴平均池化得 $\bar{d}_l^t$。
- **自适应加权块**：将 $L$ 个尺度动态特征沿通道拼接，经 $1\times1$ 卷积降维后用 Softmax 得到权重矩阵 $w_l^t \in \mathbb{R}^{h\times w}$，按空间位置自适应加权：$x_{\text{motion}}^t = \sum_l w_l^t * \bar{d}_l^t$。浅层关注细微运动（眉毛/脸颊），深层关注大幅运动（嘴部）。
- 最终与时序无关的顶层静态特征相加：$G^t = x_{\text{motion}}^t + x_L^t$。

### HSR 模块（Hierarchical Spatio-temporal AU Relationship Modelling）
- **Stage 1 局部关系建模**：将 $G^t$ 沿高度切分为上（眉毛/眼）、中（鼻/颊）、下（嘴/下巴）三个略有重叠的区域子特征。每个AU通过专属特征提取器（AFE：Conv1×1 + GAP）从其对应区域提取局部感知特征 $v_n^t$。训练时附加辅助分支，以 AU 组合预测（one-hot）监督各区域特征，强制编码区域内部AU依赖。
- **Stage 2 跨区域关系建模**：将 Stage 1 预测激活的 AU 节点，通过 GAT 层自适应连接不同面部区域的激活 AU，建模跨区时空依赖，输出最终层级关系感知特征 $\hat{V}^t$。

### 损失函数
- **AU 主损失**：非对称损失（Asymmetric Loss），缓解标签不平衡（大量负样本）：$\mathcal{L}_{AU} = -\sum_{n,t} w_n[y\log(p) + p(1-y)\log(1-p)]$。
- **辅助区域损失**：交叉熵 $\mathcal{L}_{sub}$ 监督各区域AU组合预测。
- **总损失**：$\mathcal{L} = \mathcal{L}_{AU} + \lambda \mathcal{L}_{sub}$（$\lambda=0.01$）。

## 实验与结果
- **数据集**：BP4D（328视频，~14万帧，12 AU）、DISFA（27序列，13万帧，8 AU），subject-independent 三折交叉验证。
- **评估指标**：帧级 F1 分数。
- **BP4D 结果**：ResNet-50 主干达 **66.6%**，Swin-B 主干达 **66.1%**，均为最优（Table 1）。相对最强静态基线 ME-GraphAU 提升 **1.1%**（ResNet）/ **0.6%**（Swin）；相对最强时空基线 WSRTL 提升 **0.7%**。
- **DISFA 结果**：ResNet-50 主干达 **64.9%**，Swin-B 主干达 **66.2%**，均为最优（Table 2）。相对 ME-GraphAU 提升 **1.8%**（ResNet）/ **3.1%**（Swin）。
- **消融实验**（BP4D，ResNet-50）：
  - 基础模型 F1=63.3%；MFD 贡献 **+1.3%**；HSR 全模块贡献至 65.7%；+TCN 后达 **66.6%**。
  - MFD 中自适应加权（AW）较简单相加/拼接策略各提约 0.2–0.4%（64.9→65.3%）。
  - HSR 中跨区域激活AU连接策略优于完全连接等替代方案。

## 相关工作脉络
1. **静态图像AU识别**（EAC-Net, JAA-Net, ARL, SMA-Net）：仅利用单帧全局/局部特征，无时序建模，本文明确超越此类方法。
2. **静态AU关系建模**（SRERL, FAUDT, FAN-Trans, ME-GraphAU）：用Transformer或图网络统一建模AU间关系，未考虑层级空间分布；本文 HSR 的两阶段设计更贴合面部解剖结构。
3. **时空方法**（STRAL, HSTR-Net, KS, WSRTL）：采用LSTM/ST-GNN/Transformer建模时序，但未考虑多尺度动态异质性；本文 MFD 补全了这一缺失。
4. **时序差分网络**（TDN, Yang et al. [56]）：仅单一空间尺度提取动态；本文扩展至多尺度并引入自适应加权。
5. **基于光流/辅助任务的方法**（WSRTL）：通过辅助任务编码动态；本文直接从骨干网络多尺度特征差分工效高效地建模动态。

## 局限性与未来方向
1. 面部区域切片策略（等分高度为7层后固定划分）可能不够精细，可探索更自适应的区域划分方式。
2. HSR 中跨区图边构建仅基于当前帧激活状态，可引入更先进的动态图边学习策略。
3. 时序窗口固定为 $k=1$（前后各1帧），未系统分析更大窗口或变长序列的影响。
4. 未在跨被试、跨表情类型等更极端泛化设置下验证鲁棒性。

## 研究启发与可借鉴点
1. **多尺度动态建模思想可迁移**：MFD 的"骨干网络多层特征差分 + 自适应加权融合"策略可推广至其他视频理解任务（如微表情识别、动作识别），尤其适用于目标活动在空间尺度上异质明显的场景。
2. **层级关系建模的两阶段范式**：先局部后全局的分层关系建模思路，不仅适用于 AU，也可用于多标签分类、关键点关联等存在空间层次结构的任务。
3. **辅助分支编码区域依赖**：训练时引入区域级组合预测辅助任务（Aux），可有效正则化局部特征学习，这一自监督信号设计简单且高效。
4. **非对称损失处理极度类别不平衡**：AU 识别中正样本占比极低，非对称损失的应用效果显著，可作为后续研究的标准配置。

## 关键术语表
**Action Units (AUs)**：FACS系统中描述基本面部肌肉运动的原子级单元，共34种，用于客观量化面部表情。
**Multi-scale Facial Dynamic Modelling (MFD)**：通过骨干网络多层特征差分提取多尺度面部动态，并以自适应权重融合的关键模块。
**Hierarchical Spatio-temporal AU Relationship Modelling (HSR)**：分两阶段（局部区域→跨区域）建模AU时空关系的模块。
**Adaptive Weighting Block**：MFD中的加权机制，通过学习空间位置相关的权重矩阵，针对不同AU自适应选择最有用的动态尺度。
**Temporal Convolution Networks (TCN)**：用于建模AU预测序列的时序卷积网络，替代RNN以捕获长程时序依赖。
**Asymmetric Loss**：针对多标签极度不平衡问题设计的非对称交叉熵损失，对负样本动态降权。
**Cross-regional AU Relationship Modelling (CRM)**：HSR的第二阶段，通过GAT将不同面部区域的激活AU节点连接建模跨区依赖。
**Swin-Transformer**：层次化Transformer backbone，在本文中被用作更强大的特征提取器并同样达到SOTA。

## 可复现要素
- **数据集**：BP4D、DISFA（公开数据集，需申请访问）
- **代码**：已开源，https://github.com/CVI-SZU/MDHR
- **权重**：论文未明确提及是否公开预训练权重
- **关键超参**：$k=1$（相邻帧数）、$\lambda=0.01$（辅助损失权重）、初始学习率 0.0001（cosine decay）、AdamW（$\beta_1=0.9, \beta_2=0.999$）、输入分辨率 224×224、Backbone 使用 ImageNet 预训练的 ResNet-50 / Swin-B
