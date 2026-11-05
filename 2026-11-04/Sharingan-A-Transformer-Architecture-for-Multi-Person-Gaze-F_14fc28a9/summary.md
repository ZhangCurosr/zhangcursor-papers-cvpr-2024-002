---
title: "Sharingan-A-Transformer-Architecture-for-Multi-Person-Gaze-F"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Tafasca_Sharingan_A_Transformer_Architecture_for_Multi-Person_Gaze_Following_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 08:34:50"
field: "多人生注视估计"
keywords: ["gaze following", "multi-person", "transformer", "vision transformer", "dense prediction", "conditional DPT"]
innovations: ["gaze token机制：将人物头部信息压缩为单个带位置感知的token与场景token交互", "Conditional DPT解码器：多尺度per-person条件化密集预测解码机制"]
benchmarks: ["GazeFollow", "VideoAttentionTarget", "ChildPlay"]
---

# 论文速读：Sharingan: A Transformer Architecture for Multi-Person Gaze Following

## 一句话总结
本文提出 Sharingan，一种基于 Transformer 的多人生注视跟随架构，通过引入"gaze token"机制将人物头部信息编码为单个可控 token，与场景 image token 在统一注意力框架下交互，实现单次前向传播同时预测多人注视目标，在 GazeFollow、VideoAttentionTarget、ChildPlay 三大基准上均取得 SOTA 结果。

## 研究问题与动机
1. **多人生注视效率瓶颈**：传统双塔（2-tower）CNN 架构需对每人都单独执行一次前向传播，在多人场景中极度低效，且难以扩展到多模态输入（深度、姿态等）。
2. **已有 Transformer 方法的可评估性问题**：Tonini [38]、Tu [39] 等基于 DETR 式 set prediction 的方法使用固定可学习 embedding 解码，需要事后匹配步骤才能将预测与标注对应，难以用现有 benchmark 可靠量化评估，也难以集成到更大的行为理解系统中。
3. **解码机制被忽视**：多数工作聚焦于人物表征和场景显著性建模，但对最终 gaze heatmap 的解码模块设计缺乏关注，常用低维（如 7×7）卷积+转置卷积，丢失细粒度空间信息。
4. **任务 formulation 的保持需求**：原有任务定义中"给定目标人物，预测其注视点"是核心，而 DETR 式方法同时预测 head box 和 gaze target，偏离了原始问题设定。

## 核心贡献（创新点）
1. **gaze token 机制**：将人物头部信息压缩为单个带位置感知的光注视 token，与场景 token 在同一 Transformer 中交互；与以往方法（head mask embedding、head crop tokens、gaze cone tokens）的本质区别在于无需增加 token 数量且性能更优。
2. **Conditional DPT 解码器**：提出的多尺度轻量级解码机制，从 Encoder 不同层抽取 image/gaze token 特征，在多个分辨率上逐步融合；与标准 DPT [28] 的本质区别在于引入了 per-person 条件化（conditioned on each person），使其适用于密集预测的 gaze heatmap 生成。
3. **保持原始任务 formulation 的多人生注视架构**：不重新发明 head detection，而是依赖现成检测器获取 head box，专注解决 gaze 预测；与 Tonini [38]、Tu [39] 同时预测 head box 和 gaze target 的方法本质不同，保持了原问题的可控性和可评估性。
4. **跨数据集强泛化能力**：仅在 GazeFollow 上训练，未经 fine-tune 即在 VideoAttentionTarget（Dist 0.113 vs 0.134）和 ChildPlay（0.109 vs 0.134）上大幅优于多人大模型 baseline，证明架构泛化性。

## 方法详解
- **Image Tokens**：输入场景图像 I ∈ R^(H×W×C) 经 patch projection P_img 生成 image tokens x^img ∈ R^(N×D)，附加位置编码，遵循标准 ViT 方式。
- **Gaze Tokens 生成（单人）**：头部裁剪 h_crop 经 gaze backbone（预训练 ResNet-18）产生 gaze embedding g^emb；（i）通过 gaze regressor（MLP）O_gv 预测归一化 2D 注视向量 g_v = O_gv(g^emb)，用 angular loss 监督；（ii）投影到 token 维度 x^emb = P_gaze(g^emb) ∈ R^D；头 bounding box h_bbox 经线性投影得 x^bbox = P_bbox(h_bbox) ∈ R^D；最终 location-aware gaze token：x^g = x^emb + x^bbox ∈ R^D。
- **Multi-person 扩展**：Np 人产生 Np 个 gaze tokens x^g = x^g_1 ⊕ ... ⊕ x^g_Np，concat 到 image tokens 后送入标准 ViT Encoder（L 层）。
- **Conditional DPT Decoder**：从 Encoder 第 4、8、16、32 层抽取 intermediate representations，将 image tokens 重组为 (H/k, W/k) 分辨率的特征图，与对应 gaze token 做 element-wise dot-product，产生 per-person 特征图；经 fusion module（residual convnet + 上采样）逐步融合至 (H/2, W/2) 分辨率，最终 conv head 输出 (B, Np, 1, H_hm, W_hm) 的 gaze heatmap。
- **In-Out 预测**：MLP 分类头，输入拼接的 input/output gaze token，预测 person 是否注视画面内（binary label）。
- **损失函数**：L = λ_reg * L_reg（MSE heatmap loss）+ λ_ang * L_ang（angular loss = 1 - <g_v^gt, g_v^pred>）+ λ_io * L_io（binary cross-entropy）。λ_reg = 1000，λ_ang = 3。

## 实验与结果
- **数据集**：GazeFollow（~130K 标注实例，122K 图像）、VideoAttentionTarget（164K 实例，71K 帧）、ChildPlay（257K 实例，120K 帧）。
- **评估指标**：AUC、Avg. Distance、Min. Distance、PLAH（Precision of looking at people's heads）、AP。
- **GazeFollow 主要结果**：AUC=0.944，Avg. Dist=0.113，Min. Dist=0.057（SOTA，multi-person 类别最优）；相比 Jin [16] 提升 Avg. Dist 0.013、Min. Dist 0.02。
- **VideoAttentionTarget**：Dist=0.107，PLAH=0.748；未 fine-tune 即达 0.113，优于 [16]（0.134）和 [35]（0.109）。
- **ChildPlay**：Dist=0.106，PLAH=0.600，AP=0.990（SOTA）。
- **消融结论**：location-aware gaze token 优于 head mask embedding（0.117→0.113）、head crop tokens（0.138）、gaze cone tokens（0.133）；Conditional DPT 优于 token-to-heatmap（0.302）、dot-product（0.120）、up & dot-product（0.116）；angular loss 对 heatmap 性能无显著影响但有助于 in-vs-out 预测。

## 相关工作脉络
1. **GazeFollowing [29]**：任务开创性工作，定义"in 图像中预测目标人物注视的 2D 像素位置"，无约束假设，适用于任意场景。本文在其 formulation 基础上扩展至多人生场景。
2. **Chong [7]**：引入 in-vs-out 标签和更多评估指标（包括 VideoAttentionTarget 数据集）；本文沿用其评估体系并扩展 PLAH 指标。
3. **Tonini [38] / Tu [39]**：基于 DETR 的多人生注视方法，同时预测 head box 和 gaze target；本文与其定位差异在于保持原始任务 formulation、无需匹配步骤、可可靠评估。
4. **Jin [16]**：首个卷积式多人生注视方法，但逐人独立计算忽略人物间交互；本文共享 scene 表征并通过 attention 机制建模交互。
5. **Fang [11] / Gupta [12]**：多模态（I+D+E/P）单人生注视 SOTA；本文在纯图像（I）单模态设定下超越多数方法，且支持多人生同时预测。
6. **DPT [28]**：dense prediction 的 vision transformer 解码器；本文 Conditional DPT 在其基础上引入 per-person conditioning，适配 gaze heatmap 生成。

## 局限性与未来方向
- **可解释性有限**：scene 和 person token 共享同一组 weights 处理，难以清晰分离两路信息的融合机制；未来可探索解耦（disentangle）scene/person 处理并选择性融合。
- **视频稳定性待提升**：当前架构在视频场景下的预测稳定性未充分验证，分离处理可能有助于时序一致性。
- **多模态扩展待探索**：目前仅使用图像模态（I），depth、pose 等模态的融合潜力未在本文中系统评估。
- **2D point 回归的局限**：讨论中指出直接回归 (x, y) 坐标在多峰分布场景下会收敛到各 mode 的加权平均，导致 min distance 较差，heatmap 输出更适合表达不确定性。

## 研究启发与可借鉴点
1. **单一可控 token 编码人物信息**：将人物 head crop + bbox 压缩为单个 gaze token 而非视觉 attention map 或 gaze cone，在 transformer 框架下更高效且性能更优；该思路可迁移至其他多人交互理解任务（如 joint attention、mutual gaze）。
2. **Conditional DPT 解码设计**：将 dense prediction 的 feature pyramid 思想与 per-object conditioning 结合，适用于任何"给定 query 特征 + 场景特征 → 输出 spatially dense prediction"的任务。
3. **训练/推理人数不对称策略**：训练时固定 Np^tr（如 2）以平衡计算效率，推理时处理 Np^i（全部人数），且 loss 仅从标注人物回传——这一设计巧妙解耦了训练约束与推理灵活性。
4. **Heatmap vs 2D Point 目标的系统性分析**：论文详细讨论了两种输出形式的优劣，指出 avg distance 在多标注点场景下可能误导评价；这一分析框架对其他空间预测任务有参考价值。

## 关键术语表
- **Gaze Following**：在图像中预测目标人物注视的 2D 像素位置，无需眼动仪或姿态假设。
- **Gaze Token**：将人物头部信息（embedding + bounding box）编码为单个 token，与 scene token 共同在 Transformer 中交互。
- **Conditional DPT**：本文提出的多尺度 gaze 解码器，在多个分辨率上对每人的特征进行条件化融合，输出 gaze heatmap。
- **In-vs-Out 预测**：判断人物视线是否落在图像画幅内的二分类任务。
- **PLAH（Precision of Looking at People's Heads）**：评估注视语义精度的指标，预测点和标注点落在同一 head box 内即视为正。
- **Angular Loss**：监督归一化 2D 注视向量的损失，最大化预测与真实 gaze vector 的余弦相似度。
- **ViT（Vision Transformer）**：将图像划分为 patch 并作为 token 序列输入标准 Transformer 的视觉骨干网络。
- **DETR-style Set Prediction**：将检测/预测任务建模为集合预测，使用可学习 query 和 Hungarian matching，本文不采用此范式。

## 可复现要素
- **数据集**：GazeFollow、VideoAttentionTarget、ChildPlay（均为公开数据集）。
- **代码/权重**：论文声明"code, checkpoints, and data extractions will be made publicly available soon"（截至发表时未立即开源）。
- **关键超参**：输入分辨率 224×224，输出 heatmap 64×64；gaze backbone 为 Gaze360 预训练的 ResNet-18；Transformer encoder 为 ViT-base，初始化为 Multimae [2] 权重；训练 20 epochs（GazeFollow），fine-tune 2 epochs；AdamW，lr=3e-5（cosine schedule）；λ_reg=1000，λ_ang=3；Np^tr=2。
