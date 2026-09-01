---
title: "Sharingan-A-Transformer-Architecture-for-Multi-Person-Gaze-F"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Tafasca_Sharingan_A_Transformer_Architecture_for_Multi-Person_Gaze_Following_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 20:15:47"
---

# 论文速读：Sharingan-A-Transformer-Architecture-for-Multi-Person-Gaze-F

## 一句话总结
本文提出了 Sharingan，一种基于 Transformer 的原生多目标注视跟随网络，通过引入位置感知的单 Gaze Token 与条件多尺度解码器（Conditional DPT），在保持原始任务设定（已知头部位置）的前提下实现单步前向传播同时预测任意人数注视，在 GazeFollow、VideoAttentionTarget 与 ChildPlay 三个基准上均取得 SOTA。

## 研究问题与动机
1. **单目标 CNN 的效率瓶颈**：传统双塔架构需对每张图像执行多次前向传播（每人一次），随着场景中人数增加计算成本线性增长，难以部署于实时系统。
2. **已有 Transformer 多目标方法的评估与集成缺陷**：Tonini [38] 与 Tu [39] 等方法采用固定可学习 query 集同时解码头部框与注视目标，属于集合预测范式，推理后需额外匹配步骤才能与标注对齐，无法直接复用现有注视跟随基准，也不易嵌入下游行为理解管线。
3. **解码模块设计被严重低估**：既往工作过度聚焦于人物表征与场景显著图构建，对最终热图生成环节仅使用低维卷积+反卷积，输入特征维度常低至 7×7，丢失了细粒度空间信息。
4. **任务形式应保持可追溯性**：多目标扩展不应改变“给定人物→预测注视点”的原始 formulation，否则难以与单目标文献对齐评估协议。

## 核心贡献（创新点）
1. **单 Gaze Token 位置感知表征**：将头部裁剪经轻量骨干压缩为单一 embedding，再与边界框投影嵌入相加构成唯一 Gaze Token；与使用视觉注意力图或注视锥体的方案本质不同，大幅减少冗余 token 数量且实验证明更利于 Transformer 交互。
2. **条件多尺度解码器（Conditional DPT）**：将 DPT 密集预测范式改造为多人条件化版本，从编码器多层抽取特征并按模拟分辨率逐级融合，通过点积注入人物专属条件，比直接 MLP 回归或一次性点积方案能更好捕捉多峰分布与不确定性。
3. **原生支持任意人数的统一架构**：打破双塔多 pass 限制，Scene Token 与 Gaze Token 在标准 ViT Encoder 中联合交互，推理时只需一次前向传播即可输出所有人员的注视热图。
4. **三任务联合优化与强跨域泛化**：联合热图 MSE、角度向量余弦损失与 In-Out 二分类 BCE，并在零样本跨数据集与微调实验中验证，各项指标全面超越既有单目标与多目标方法。

## 方法详解
- **Image Token 生成**：输入场景图 $\mathbf{I} \in \mathbb{R}^{H \times W \times C}$ 经 Patch Projection $\mathcal{P}_{img}$ 切块并添加位置编码，得到 $\mathbf{x}^{img} \in \mathbb{R}^{N \times D}$。
- **Gaze Token 构建**：头部裁剪 $\mathbf{h}_{crop}$ 输入预训练 ResNet-18（Gaze360 权重）得到 $\mathbf{g}^{emb}$；$\mathbf{g}^{emb}$ 经线性投影 $\mathcal{P}_{gaze}$ 映射至 token 维度，同时头部 bbox $\mathbf{h}_{bbox}$ 经 $\mathcal{P}_{bbox}$ 投影为位置嵌入，两者相加得最终 Token：$\mathbf{x}^g = \mathcal{P}_{gaze}(\mathbf{g}^{emb}) + \mathcal{P}_{bbox}(\mathbf{h}_{bbox})$。
- **Transformer Encoder**：拼接 $\mathbf{x}^{img}$ 与 $N_p$ 个 $\mathbf{x}^g$ 组成 $\mathbf{x} \in \mathbb{R}^{N_t \times D}$，通过 $L$ 层标准 ViT Block 进行全局自注意力交互，输出 $\mathbf{x}^{out}$。
- **Conditional DPT Decoder**：取编码器第 $l_k \in \{4, 8, 16, 32\}$ 层的中间表示，重建为 $(H/k, W/k)$ 的图像状特征；将特征图复制 $N_p$ 次后与每个 Gaze Token 做逐元素点积实现条件过滤，经残差卷积融合、上采样至 $H/2 \times W/2$，最后通过卷积 Head 压缩通道并Resize 到 $H_{hm} \times W_{hm}$，输出形状为 $(B, N_p, 1, H_{hm}, W_{hm})$ 的注视热图集合。
- **In-Out 分类头**：拼接输入与输出 Gaze Token，经 7 层 MLP 输出每人是否在画面内注视的二分类 logits。
- **损失函数**：$\mathcal{L} = \lambda_{reg}\mathcal{L}_{reg} + \lambda_{ang}\mathcal{L}_{ang} + \lambda_{io}\mathcal{L}_{io}$，其中 $\mathcal{L}_{reg}$ 为预测与 GT 热图的像素级 MSE，$\mathcal{L}_{ang} = 1 - <\mathbf{g}_v^{gt}, \mathbf{g}_v^{pred}>$ 监督归一化注视方向向量，$\mathcal{L}_{io}$ 为标准二元交叉熵。

## 实验与结果
- **数据集**：GazeFollow（~130K 实例）、VideoAttentionTarget（164K 实例）、ChildPlay（257K 实例）。
- **评估指标**：AUC、Avg. Dist.、Min. Dist.、AP、PLAH（预测与标注 gaze 点落在同一头部框内计为正）。
- **主要结果**：GazeFollow 测试集 AUC=0.944、Avg. Dist.=0.113、Min. Dist.=0.057；VideoAttentionTarget Dist.=0.107、PLAH=0.738、AP=0.891；ChildPlay Dist.=0.106、PLAH=0.600、AP=0.990，三项基准均刷新 SOTA。
- **关键对比提升**：相对唯一可公平比较的多目标基线 Jin [16]，在 GazeFollow 上 Avg. Dist. 提升 0.013、Min. Dist. 提升 0.02；未微调的 zero-shot 模型在 VideoAttentionTarget 上 Dist. 达 0.113，优于竞品 0.021，泛化能力突出。
- **消融结论**：Location-aware Gaze Token 最优（AUC 0.944 vs 头掩码 0.940）；Conditional DPT 显著优于直接 MLP（AUC 0
