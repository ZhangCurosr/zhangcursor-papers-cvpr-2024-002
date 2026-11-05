---
title: "Multi-scale-Dynamic-and-Hierarchical-Relationship-Modeling-f"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_Multi-scale_Dynamic_and_Hierarchical_Relationship_Modeling_for_Facial_Action_Units_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 00:43:16"
field: "面部动作单元识别"
keywords: ["facial action units", "action unit recognition", "multi-scale temporal differencing", "hierarchical relationship modeling", "spatio-temporal modeling", "graph attention network"]
innovations: ["多尺度自适应面部动态建模模块，首次区分不同AU激活的尺度异质性", "两阶段层级时空关系建模，分别处理局部区域与跨区域AU依赖", "在BP4D和DISFA上取得面部动作单元识别新SOTA"]
benchmarks: ["BP4D", "DISFA"]
---

# 论文速读：Multi-scale Dynamic and Hierarchical Relationship Modeling for Facial Action Units Recognition

## 一句话总结
本文提出多尺度动态与层级关系建模（MDHR）方法，通过多尺度时序差分网络自适应捕捉不同空间尺度的面部动态，并采用两阶段策略分层建模AU之间的局部与跨区时空关系，在BP4D和DISFA数据集上取得面部动作单元识别的新SOTA。

## 研究问题与动机
- **问题1：忽略AU层级空间关系** — 现有Transformer和图方法对AU对采用统一策略建模，未考虑同区域/邻近区域AU因共享面部肌肉而关联更强的自然层级结构。
- **问题2：忽略面部动态异质性** — 不同AU激活在运动范围和幅度上存在显著差异（如AU25为大范围嘴部变形，AU2为微妙眉周运动），既有方法未区分不同空间尺度的动态贡献。
- **静态方法的局限** — 基于单帧图像的方法忽略时序动态，基于patch的方法忽略上下文信息且易受 landmark 检测误差影响。
- **现有时序方法的不足** — LSTM、3D CNN等对细微肌肉运动不敏感；光流等方法未考虑不同AU动态特征在不同空间尺度的差异化重要性。

## 核心贡献（创新点）
- **首次提出多尺度自适应面部动态建模模块（MFD）** — 与现有方法本质区别在于同时捕获粗细粒度动态并按AU需求自适应加权，而非统一处理所有空间尺度。
- **首次分层建模AU时空关系（HSR）** — 两阶段策略分别处理同区域局部关系与跨区域全局关系，区别于现有方法对AU对采用无差别建模方式。
- **达到新SOTA性能** — 在BP4D和DISFA基准上超越所有现有静态、图关系及时空方法，MFD与HSR模块互补贡献验证了设计有效性。

## 方法详解
**整体架构（MDHR）：**
- 输入T帧连续视频，取中心帧$f^t$及其$k$帧邻域帧（本文k=1），经Backbone提取L层多尺度静态特征$X_l$（$l=1,...,L$）。
- 依次经过MFD模块生成时空全局特征$G^t$，再经HSR模块生成分区感知AU特征$\hat{V}^t$，最后用TCN+相似度计算预测N个AU。

**MFD模块（多尺度面部动态建模）：**
1. **多尺度时序差分**：对相邻帧同层特征做逐点差分$d_l^t = x_l^t - x_l^{t-1}$，得到L组不同尺度动态特征，经Conv2D统一形状后沿时序轴平均池化得到$\bar{d}_l^t$。
2. **自适应加权**：拼接所有尺度特征→1×1卷积降维→Softmax归一化得到权重矩阵$w_l^t$，对动态特征加权融合：$x_{motion}^t = \sum_l w_l^t * \bar{d}_l^t$。
3. **静态-动态融合**：与输出层静态特征相加得$G^t = x_{motion}^t + x_L^t$。
4. **关键设计**：浅层对应微妙运动（如眉毛），深层对应大范围运动（如嘴部），加权机制使不同AU自动关注最相关尺度。

**HSR模块（层级时空AU关系建模）：**
1. **局部关系建模（第一级）**：
   - 将$G^t$沿高度切分为上/中/下三个重叠区域（高度7等分，各取4/7高度）。
   - 每个AU通过专属特征提取器（1×1卷积+GAP）从其所属区域特征中学习，编码空间邻近AU信息。
   - 训练时加辅助分支预测区域内AU组合（one-hot），约束网络学习局部依赖。
2. **跨区域关系建模（第二级）**：
   - 以局部特征为节点，基于第一阶段AU预测激活状态构建图边（激活AU对其他AU影响更大）。
   - 使用GAT层建模跨区域依赖：边权重$e_{n,m}^t$由注意力机制学习，聚合得到最终AU特征$\hat{v}_n^t$。

**损失函数：**
- 主损失：非对称损失（Eq.10），缓解负样本（未激活AU）远多于正样本的不平衡问题。
- 辅助损失：区域AU组合预测的交叉熵（Eq.11）。
- 总损失：$\mathcal{L} = \mathcal{L}_{AU} + \lambda\mathcal{L}_{sub}$，$\lambda=0.01$。

## 实验与结果
**数据集：** BP4D（328视频，14万帧，12 AU标注）、DISFA（27视频序列，13万帧，8 AU标注）。

**评估指标：** Frame-based F1 score，subject-independent 3-fold交叉验证。

**SOTA对比（BP4D）：**
- ResNet-50骨干：平均F1 **66.6%**，超越次优WSRTL（65.9%）+0.7%；优于静态关系SOTA ME-GraphAU（65.5%）+1.1%。
- Swin-B骨干：平均F1 **66.1%**，表现稳健。
- 部分AU显著提升：AU14达69.5%（新SOTA），AU25达88.2%。

**SOTA对比（DISFA）：**
- ResNet-50骨干：平均F1 **64.9%**，超越次优WSRTL（64.6%）+0.3%。
- Swin-B骨干：平均F1 **66.2%**，提升更明显（+1.6% over WSRTL）。
- AU1达65.4%（Swin-B），显著领先。

**消融实验关键结论：**
- MFD贡献+1.3%，HSR贡献+1.8%（相对纯静态基线63.3%）。
- 多尺度动态融合中，全尺度自适应加权（AW）最优（65.3%），优于直接相加/拼接。
- 跨区域图中"连接已激活AU"策略最优（65.1%），验证了激活状态驱动的边构建有效性。
- TCN时序建模带来+0.3%增益。

## 相关工作脉络
- **EAC-Net [19]/JAA-Net [35]**：基于landmark裁剪局部patch进行AU识别，本文通过全局特征+区域切片在保留上下文的同时避免无关区域噪声。
- **SRERL [17]/ME-GraphAU [29]**：图方法建模AU关系但未区分空间层级，本文HSR显式建模局部/跨区域两层关系。
- **FAUDT [13]/FAN-Trans [57]**：Transformer建模AU关系但对所有AU对使用相同自注意力，本文考虑同区域AU关联更强的先验。
- **STRAL [37]/HSTR-Net [47]**：时空图方法，本文MFD在多个backbone层直接计算差分而非仅处理特征级。
- **WSRTL [52]**：弱监督区域与 temporal learning，本文无额外数据集/任务，纯端到端建模。
- **TDN [56]**：单尺度时序差分网络，本文扩展为多尺度并引入自适应加权。

## 局限性与未来方向
- **面部区域划分策略**：当前固定比例切片（7等分）可能不够精细，可探索数据驱动的区域自适应划分。
- **图边学习策略**：当前基于激活状态硬连接，可引入更先进的可学习图边机制（如动态图神经网络）。
- **超参数k敏感性**：邻域帧数k对性能有影响，未做系统分析（见补充材料）。
- **跨数据集泛化**：仅在BP4D和DISFA验证，未测试于其他AU数据集或跨域场景。

## 研究启发与可借鉴点
- **多尺度动态融合范式**：MFD的"差分→统一形状→自适应加权→融合"流程可迁移至其他视频理解任务（如微表情识别、行为识别），尤其是动作幅度差异大的多标签场景。
- **层级关系建模策略**：HSR的两阶段设计（局部约束+全局聚合）可用于任何具有空间层次结构的图关系学习任务，如人体关节预测、医学图像分析。
- **辅助分支蒸馏设计**：区域AU组合预测作为辅助损失，强制网络编码局部依赖，此思路可扩展至其他多标签分类任务的层次结构正则化。
- **激活状态驱动的图构建**：基于前一阶段预测动态决定图边连接，避免全连接的计算开销，适用于节点关系稀疏的场景。
- **团队结合点**：若团队关注细粒度面部分析或视频时序建模，可借鉴MFD的多尺度差分思想与HSR的层次关系建模，应用于表情强度估计、病理表情识别等任务。

## 关键术语表
- **Action Unit (AU)**：面部动作单元，FACS系统中描述独立面部肌肉运动的最小语义单元。
- **Multi-scale Temporal Differencing**：在多个backbone层级计算相邻帧特征差分，捕获不同粒度的面部动态。
- **Adaptive Weighting Block**：学习各空间尺度动态特征的重要性权重，实现AU感知的动态融合。
- **Hierarchical Spatio-temporal Relationship**：按面部区域层级（局部→跨区域）建模AU间的依赖关系。
- **AU-specific Feature Extractor (AFE)**：每个AU专属的1×1卷积+池化层，从对应面部区域提取局部感知特征。
- **Cross-regional AU Relationship Modeling (CRM)**：基于GAT建模不同面部区域间AU的激活关联。
- **Asymmetric Loss**：针对多标签不平衡设计的损失函数，对负样本动态降权。
- **BP4D / DISFA**：两个标准的面部AU识别数据集，分别标注12和8类AU。

## 可复现要素
- **数据集**：BP4D和DISFA（公开可用）。
- **代码**：已开源，GitHub链接：https://github.com/CVI-SZU/MDHR。
- **骨干网络**：ResNet-50和Swin-B（ImageNet预训练）。
- **关键超参**：邻域帧数k=1，辅助损失权重λ=0.01，初始学习率0.0001（余弦衰减），AdamW优化器（β1=0.9, β2=0.999）。
- **输入预处理**：MTCNN检测对齐，裁剪224×224人脸区域。
- **训练策略**：subject-independent 3-fold交叉验证。
