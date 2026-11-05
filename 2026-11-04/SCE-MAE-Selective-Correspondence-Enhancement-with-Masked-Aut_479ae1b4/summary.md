---
title: "SCE-MAE-Selective-Correspondence-Enhancement-with-Masked-Aut"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Yin_SCE-MAE_Selective_Correspondence_Enhancement_with_Masked_Autoencoder_for_Self-Supervised_Landmark_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 16:37:48"
---

# 论文速读：SCE-MAE-Selective-Correspondence-Enhancement-with-Masked-Autoencoder-for-Self-Supervised-Landmark-Estimation

## 一句话总结
提出 SCE-MAE 框架，首次将区域级 SSL 方法 MAE 引入无监督面部 landmark 估计的预训练阶段，并通过 CARB 模块与局部约束排斥损失（LCR Loss）选择性精炼关键局部对应关系，在 landmark matching 与 detection 任务上大幅超越现有 SOTA。

## 研究问题与动机
- 自监督面部 landmark 估计需要学习对局部区域高度区分性的特征表示，但现有 SOTA 方法多依赖实例级 SSL（如 MOCO、BYOL）预训练，生成的初始局部特征区分度不足。
- 现有工作需构建高内存消耗的 hypercolumn 进行特征聚合，计算与存储成本高昂，且不利于密集预测任务。
- 现有方法对所有空间特征对均施加对应关系监督，忽视了非 landmark 均匀区域（脸颊、额头、背景）产生的冗余/无效对应关系，导致参数利用效率低下。
- 非 landmark 区域面积大且纹理均匀，而 landmark 区域稀疏且独特；如何通过选择性优化关键对应关系而非全量对应，是提升特征判别性的核心动机。

## 核心贡献（创新点）
1. 首次将基于掩码图像建模（MIM）的 MAE 作为第一阶段骨干用于自监督面部 landmark 估计，其“掩码-重建”任务天然契合 dense 预测需求，生成质量更高的初始局部特征。
2. 提出对应近似与精炼模块（CARB），利用 CLS token 相似度自动划分 attentive（landmark/关键区域）与 inattentive（背景/非关键区域）token，并采用密度峰值聚类对冗余非关键 token 进行代理降维。
3. 设计局部约束排斥损失（LCR Loss），解耦空间距离惩罚与对应类型权重，直接且有针对性地精炼关键局部对应关系，避免全量特征对盲目优化。
4. 在多个 benchmark 上以显著更小的参数量和更低内存开销，在 landmark matching（提升约 20%-44%）与 detection（提升约 9%-15%）任务上刷新 SOTA，并在少样本设定下展现更强鲁棒性。

## 方法详解
- **第一阶段（Stage 1）：** 在 CelebA（162,770 张）上使用 MAE（ViT 骨干）进行区域级 SSL 预训练。输入切分为 patch 并添加位置编码，随机采样二进制掩码 M 遮挡部分 patch，encoder 处理未遮挡 patch 输出特征 $\hat{f}^p$，再用 [MASK] token 填充掩码位置，lightweight decoder 以像素级 MSE 重建原始图像，迫使模型学习强判别性局部纹理与空间上下文。
- **Attentive-Inattentive 分离：** 冻结 backbone，计算 CLS token 与所有 patch token 的注意力相似度 $Sim_{cls} = \text{Softmax}(K \cdot q_{cls} / \sqrt{d}) \in \mathbb{R}^N$。按得分选取前 $\eta \cdot N$ 个 token 为 attentive group（覆盖 landmark），其余为 inattentive group。
- **Inattentive Token Clustering：** 对 inattentive tokens 应用密度峰值聚类，计算每个 token 的局部密度 $\rho_i = \exp(-\sum_{t_j \in T_{inatt}} \|t_i - t_j\|_2^2)$ 与其到更高密度 token 的最短距离 $\delta_i$，以 $\rho_i \cdot \delta_i$ 为得分选取 top-$K_c$ 个 cluster centers 作为代理，丢弃其余冗余 token。
- **CARB 与 LCR Loss：** 将保留的 attentive tokens 与 inattentive cluster centers 重组为完整 2D feature map，经轻量 projector $\Phi$ 投影后计算对应概率 $p(t_j|t_i)$。LCR Loss 为：$\mathcal{L}_{LCR} = \sum_{t_i \in T} \sum_{t_j \in T} \log(\|t_i - t_j\| + 1) \cdot \lambda_{rep}(t_i, t_j) \cdot p(t_j|t_i; \Phi, x)$。其中空间距离项饱和惩罚远距离错误对应；排斥系数 $\lambda_{rep}$ 按 att-att、att-inatt、inatt-inatt 赋予不同权重（实践中 att 相关设为 5，inatt-inatt 设为 2）。
- **推理优化：** 训练完成后推理时跳过聚类步骤，直接使用原始特征；为避免直接缩小 patch 尺寸导致计算爆炸与表征退化，采用 cover-and-stride 技术进行特征图空间扩张。

## 实验与结果
- **数据集与设置：** 预训练使用 CelebA；测试涵盖 MAFL、300W、AFLW_M、AFLW_R（含作者重新标注的 AFLW_RC）。matching 使用 1000 对图像（500 同身份/500 异身份），detection 以眼间距百分比（% inter-ocular distance）为误差指标。
- **Landmark Matching（Table 1）：** 在特征维度 64/128/256 下，SCE-MAE（DeiT-T/S/B）均以更小参数量全面超越 CL、LEAD。最强结果（DeiT-B, 256 dim）同身份误差降至 0.27，异身份降至 1.61，相对 SOTA 提升约 20%-44%。
- **Landmark Detection - 全标注（Table 2）：** 即使仅使用最小骨干 DeiT-T，亦在全部四个数据集上优于使用 hypercolumn 的基线；最佳配置（DeiT-S/B）相对 SOTA 提升约 9%-15%。重新标注的 AFLW_RC 误差普遍更低，验证了数据质量的重要性。
- **Landmark Detection - 少样本（Table 3）：** 在 1-100 张标注样本设定下，SCE-MAE 在所有特征维度下均取得最优，平均相对提升 8.6%，最高达 20.1%，且标准差更低，鲁棒性显著优于 CL/LEAD。
- **消融与可视化（Table 4, Fig 5）：** 仅用 MAE baseline 已超越 CL/LEAD；加入聚类主要提升同身份 matching；LCR Loss 显著提升跨身份 matching；t-SNE 显示本方法聚类轮廓系数（Silhouette Coefficient）最高，landmark 分布最紧凑且类间分离最清晰。
- **失败模式：** 极端姿态差异或严重遮挡时匹配性能仍会下降（见 Supplementary）。

## 相关工作脉络
- **ContrastLandmark (CL) [9]：** 基于 MOCO 预训练 + hypercolumn + 全对对应监督。本文与其本质差异在于采用 MAE 替代实例级对比学习，并用 CARB+LCR 替代昂贵的 hypercolumn 与全量对应优化。
- **LEAD [19]：** 基于 BYOL 预训练 + 对应分布对齐蒸馏。本文指出其仅依赖第一阶段特征聚合，未针对性消除冗余对应，且 hypercolumn 内存开销大。
- **DVE [46]：** 早期无监督 landmark 工作，基于等价学习（equivalence learning）与多样性损失。本文将其视为无标注 baseline，证明 SSL+MIM 路线在判别性上更具优势。
- **MAE [14] / BEiT [3] / SimMIM [50]：** 区域级掩码图像建模代表。本文将其
