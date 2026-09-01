---
title: "SCE-MAE-Selective-Correspondence-Enhancement-with-Masked-Aut"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Yin_SCE-MAE_Selective_Correspondence_Enhancement_with_Masked_Autoencoder_for_Self-Supervised_Landmark_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 15:22:31"
---

# 论文速读：SCE-MAE-Selective-Correspondence-Enhancement-with-Masked-Aut

## 一句话总结
提出SCE-MAE两阶段框架，首次将掩码图像建模（MAE）引入自监督人脸关键点估计，并通过注意力分离、密度聚类与局域约束排斥损失（LCR Loss）实现选择性对应精炼，在零标注条件下显著提升关键点匹配与检测精度。

## 研究问题与动机
- **核心问题**：在缺乏人工标注的条件下，如何学习具有强局部区分性的特征表示，以准确定位稀疏的人脸关键点。
- **预训练范式不匹配**：现有SOTA（CL、LEAD）采用实例级对比学习（MOCO/BYOL），其目标偏向类别级语义不变性，难以捕捉稠密预测任务所需的细粒度局部纹理差异。
- **特征构建昂贵且冗余**：前作依赖计算与内存开销巨大的hypercolumn，并对所有空间特征对无差别地建立对应关系，忽略了人脸关键点稀疏、非关键点区域（脸颊/额头/背景）均匀分布的先验，导致参数利用率低下。
- **动机假设**：MIM协议通过从有限上下文重建被掩码区域，能更自然逼出关键点区域的判别性特征；若能识别重要区域并仅精炼关键对应关系，可大幅提升表示质量并降低计算负担。

## 核心贡献（创新点）
1. **首次将MAE区域级SSL引入自监督人脸关键点估计**。与CL/LEAD的实例级对比学习本质不同，MAE的重建任务天然契合关键点稀疏分布特性，输出更具区分度的初始局部特征。
2. **提出对应近似与精炼模块（CARB）与局域约束排斥损失（LCR Loss）**。通过CLS相似度划分注意力/非注意力token，利用密度峰值聚类压缩非关键点表征，LCR Loss结合空间距离与对应类型加权惩罚错误强对应，实现“选择性”精炼而非全量优化。
3. **系统性突破SOTA且显著降低存储/计算开销**。摒弃昂贵的hypercolumn，仅基于原始ViT输出特征；在MAFL、300W、AFLW等数据集上，关键点匹配误差降低约20%-44%，检测误差降低约9%-15%，且在有限标注下表现更稳健。

## 方法详解
- **两阶段架构**：第一阶段使用MAE在CelebA上进行区域级SSL预训练（ViT backbone），第二阶段冻结骨干，训练轻量级projector进行对应精炼。
- **注意力/非注意力分离（Attentive-Inattentive Separation）**：计算CLS token与所有patch token的相似度 $Sim_{cls} = \mathrm{Softmax}(K \cdot q_{cls} / \sqrt{d})$。按相似度排序，取top $\eta \cdot N$ 为attentive tokens（覆盖关键点及重要面部区域），其余为inattentive tokens（覆盖脸颊、额头、背景等均匀区域）。
- **非注意力Token聚类（Inattentive Token Clustering）**：对inattentive tokens应用密度峰值聚类算法，定义密度 $\rho_i = \exp(-\sum \|t_i-t_j\|_2^2)$ 与相对距离 $\delta_i$，以得分 $\rho_i \cdot \delta_i$ 选取top-$K_c$个作为cluster centers。丢弃其余非注意力tokens，仅用聚类中心代理原特征，大幅减少后续需优化的对应对数量。
- **对应近似与精炼模块（CARB）与LCR Loss**：将丢弃的tokens替换为cluster centers后拼合为完整2D feature map，送入projector。设计损失：
  $\mathcal{L}_{LCR} = \sum_{t_i} \sum_{t_j} f_{loc}(t_i, t_j) \cdot \lambda_{rep}(t_i, t_j) \cdot p(t_j|t_i; \Phi, x)$
  其中 $f_{loc}=\log(\|t_i-t_j\|+1)$ 为空间距离衰减系数，$p(t_j|t_i)$ 为softmax对应概率，$\lambda_{rep}$ 为对应类型权重（$r_{att-att}$与$r_{att-inatt}$设较高值以优先区分关键点与跨类边界，$r_{inatt-inatt}$设较低值）。该损失系统性地弱化空间遥远且类型不匹配的强对应，仅直接精炼重要对应关系。
- **推理优化（Inference）**：推理时跳过聚类与近似步骤，直接使用原始特征。为避免直接缩小patch size带来的显存与性能代价，采用cover-and-stride技术将特征图空间扩展至图像分辨率，获得细粒度表示。

## 实验与结果
- **数据集与设置**：在CelebA（162,770张）上预训练（DeiT-T/S/B，400 epochs，batch 512，lr 3e-4，patch 8）。测试集包括MAFL、300W、AFLW_M及重新标注的AFLW_R（AFLW_RC）。评估指标为Mean Pixel Error（匹配）与Inter-ocular Distance百分比（检测）。
- **关键点匹配**：与CL、LEAD在相同特征维度下对比，SCE-MAE（DeiT-T仅5.4M参数）已超越前者；DeiT-B模型在相同身份（Same）与不同身份（Diff）匹配上分别达到0.27与1.61像素误差，较SOTA提升约20%-44%。
- **关键点检测（全标注）**：即使使用最小骨干DeiT-T且无需hypercolumn，在各数据集上误差均低于CL/LEAD；最佳模型在4个基准上平均降低误差约9%-15%。重新标注的AFLW_RC证实了基线结果的注释质量问题。
- **关键点检测（有限标注）**：在AFLW_M上仅用1-100张标注样本实验，SCE-MAE平均相对增益8.6%，最高达20.1%，且标准差更低，证明特征学习更稳健。
- **消融与可视化**：Baseline（仅MAE骨干特征）已优于CL/LEAD的hypercolumn；引入聚类主要提升Same-ID匹配，LCR Loss主要提升Diff-ID匹配；t-SNE显示SCE-MAE的聚类内聚性与类间分离性最佳（Silhouette Coefficient最高）。

## 相关工作脉络
- **对比学习SSL vs 掩码图像
