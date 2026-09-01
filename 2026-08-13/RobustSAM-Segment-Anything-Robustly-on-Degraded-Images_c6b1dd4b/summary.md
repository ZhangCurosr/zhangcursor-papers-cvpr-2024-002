---
title: "RobustSAM-Segment-Anything-Robustly-on-Degraded-Images"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_RobustSAM_Segment_Anything_Robustly_on_Degraded_Images_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 15:22:28"
---

# 论文速读：RobustSAM-Segment-Anything-Robustly-on-Degraded-Images

## 一句话总结
本文提出 RobustSAM，通过在冻结的原始 SAM 主干中插入轻量级抗退化 Token 与掩码特征生成模块，使模型在保持零样本迁移能力的同时显著提升对多种合成及真实图像退化的分割鲁棒性，并同步构建开源了 68.8 万对多退化类型的 Robust-Seg 数据集与统一评测协议。

## 研究问题与动机
- SAM 在低光照、噪声、模糊、恶劣天气等退化条件下性能显著下降，且错误的掩码会直接拖累依赖其作为结构先验的下游任务（如去 haze、去模糊）。
- 现有“先复原后分割”的两阶段方案多面向人类视觉感知优化，无法保证复原后的特征更能被 SAM 等分割模型利用。
- 直接微调 SAM 整体或解码器极易破坏其预训练的零样本泛化能力，并引发灾难性遗忘。
- 缺乏涵盖多类退化且保留语义掩码的大规模训练/评测基准，阻碍了该方向的系统性推进。

## 核心贡献（创新点）
- 提出 AOTG 与 AMFG 双模块架构：在完全冻结 SAM 图像编码器、提示编码器与原解码器的前提下，仅新增约 403 MB 可学习参数，实现退化不变特征提取。
- 设计特征一致性蒸馏机制：通过对比退化输入与原始清晰输入的特征输出，强制模型学习不受退化干扰的结构化表示，避免对原模型知识的破坏。
- 引入频域退化抑制策略：将特征转换至频域后分离幅值（退化风格）与相位（结构信息），利用 1×1 卷积过滤幅值成分以剥离退化干扰。
- 构建 Robust-Seg 数据集：整合 7 个公开数据集并施加 15 种合成退化（含恒等映射保底），形成 68.8 万对高质量图像-掩码样本，填补退化分割基准空白。

## 方法详解
- **模型整体流程**：训练时 SAM 核心组件全部冻结；退化图像经图像编码器后，原 SAM 输出令牌微调为鲁棒输出令牌（ROT），再并行送入 AOTG 与 AMFG 模块；AMFG 输出经 Feature Fusion 模块整合后与 AOTG 输出的 $\hat{T}_{RO}$ 结合，由 MLP 生成动态分类器输出最终掩码；推理阶段仅部署训练好的 RobustSAM。
- **抗退化掩码特征生成模块（AMFG）**：输入特征并行经过实例归一化（IN）与批归一化（BN），IN 消除退化风格、BN 防止细节丢失；两路特征经注意力门控融合后与原特征通道拼接，再接入 SE 通道注意力（SEC）进行自适应加权；随后送入傅里叶退化抑制模块，在频域剔除幅值中的退化成分、保留相位结构，最终逆变换回空间域得到 $\hat{F}_{CFD}$ 与 $\hat{F}_{MFD}$。
- **抗退化输出令牌生成模块（AOTG）**：采用极简设计，将 ROT 依次通过多层 IN 与单层 MLP，快速滤除令牌中对退化敏感的噪声信息，得到 $\hat{T}_{RO}$。
- **一致性损失设计**：掩码特征一致性损失 $\mathcal{L}_{\mathrm{MFC}} = \|\hat{F}_{CFD} - F_{CF C}\|_2 + \|\hat{F}_{MFD} - F_{MFC}\|_2$，令牌一致性损失 $\mathcal{L}_{\mathrm{TC}} = \|\hat{T}_{RO} - T_{OC}\|_2$，分别约束互补特征与分类令牌向清晰输入对应的特征对齐。
- **分割损失与总损失**：$\mathcal{L}_{\mathrm{Seg}} = \mathcal{L}_{\mathrm{Dice}} + \lambda_3 \mathcal{L}_{\mathrm{Focal}}$，总损失 $\mathcal{L}_{\mathrm{Overall}} = \mathcal{L}_{\mathrm{MFC}} + \lambda_1 \mathcal{L}_{\mathrm{TC}} + \lambda_2 \mathcal{L}_{\mathrm{Seg}}$，在保障特征鲁棒性的同时优化像素级分割精度。

## 实验与结果
- **数据集与评测设置**：训练使用 MSRA10K、ThinObject-5k、LVIS 全量数据；测试覆盖 MSRA10K、LVIS（训练可见）、NDD20/STREETS/FSS-1000/COCO（零样本不可见）及 BDD-100k/LIS（真实世界退化）；基线包含 SAM、HQ-SAM、AirNet+SAM、URIE+SAM。
- **可见数据集**：MSRA10K 点提示平均 IoU 达 0.8616（优于 HQ-SAM 的 0.8373），LVIS 框提示平均 IoU 达 0.7511，退化与清晰场景均全面领先。
- **零样本 unseen 数据集**：NDD20/STREETS/FSS-1000 点提示平均 IoU 达 0.8216；COCO 框提示 AP 达 0.5130，显著超越 SAM 基线（AP 0.5002）。
- **真实世界退化数据集**：BDD-100k 与 LIS 点提示 IoU/Dice 达 0.3717/0.8926，框提示 IoU/Dice 达 0.8958/0.9
