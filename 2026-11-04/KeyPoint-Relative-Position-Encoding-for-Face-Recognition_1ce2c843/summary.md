---
title: "KeyPoint-Relative-Position-Encoding-for-Face-Recognition"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Kim_KeyPoint_Relative_Position_Encoding_for_Face_Recognition_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 08:31:37"
field: "人脸与生物特征识别"
keywords: ["KeyPoint Relative Position Encoding", "ViT", "face recognition", "affine robustness", "low-quality recognition", "relative position encoding"]
innovations: ["提出 KP-RPE 将关键点位置动态融入 ViT 相对位置编码，提升未见仿射变换鲁棒性", "设计 Relative/Multihead Relative f(P) 变体，使查询-键偏置同时依赖 patch 间距离与关键点相对位置", "在人脸与步态识别上统一验证，仅以 0.02-0.07 GFLOPs 增量取得显著收益"]
benchmarks: ["TinyFace", "IJB-S", "CFPFP", "IJB-C", "Gait3D", "WebFace4M", "WebFace12M"]
---

# 论文速读：KeyPoint-Relative-Position-Encoding-for-Face-Recognition

## 一句话总结
本文提出 KP-RPE（KeyPoint Relative Position Encoding），将关键点（如面部 landmarks）信息融入 ViT 的相对位置编码中，使模型能动态适应尺度、平移和姿态变化，从而显著提升人脸识别在低质量/未对齐图像上的鲁棒性，同时在高质量对齐数据集上保持竞争力。

## 研究问题与动机
1. **对齐失败是真实场景瓶颈**：人脸识别主流 SOTA 方法高度依赖 RetinaFace 等前端对齐，但低分辨率、弱光等低质量图像中 landmark 检测不准，导致测试时对齐失效、性能骤降。
2. **Abs-PE 对未见仿射变换极度敏感**：Toy example 显示，ViT + Absolute PE 在 MNIST 上 98.1%，在 AffNIST（未见仿射变换）上暴跌至 77.27%。
3. **既有 RPE 是静态的**：标准 RPE / iRPE 仅编码"邻近 patch 更重要"的先验，但其查询-键相对关系不随图像中语义关键点的位置变化而动态调整，因此对缩放/平移/旋转变化的适应有限。
4. **低质量人脸识别现有方案局限**：AdaFace、A-SKD 等针对低质量样本设计 loss 或蒸馏策略，但仍假设训练/推理输入已对齐，未从根本上解决未对齐输入下的空间关系建模问题。

## 核心贡献（创新点）
1. **首次将 RPE 与关键点结合**：提出 KP-RPE，使自注意力的偏置项 B_ij 依赖于图像中关键点 P 的位置，从而动态调整查询-键的空间关系；与 iRPE 的本质区别在于 RPE 不再是静态查找表，而是随输入图像中关键点分布变化。
2. **三种 f(P) 变体并验证最优设计**：Absolute f(P)、Relative f(P) 和 Multihead Relative f(P)；后两者通过引入查询/key 与关键点之间的相对网格差值 D，获得更强的表达力，后者在 unaligned 数据集上带来最大收益。
3. **极高效率**：相比 iRPE 引入约 0.71/1.42 GFLOPs 额外开销，KP-RPE 仅增加 0.02/0.07 GFLOPs，且 landmark 检测推理时间可忽略不计；比 iRPE 快一个数量级。
4. **跨任务通用性验证**：不仅在人脸（TinyFace、IJB-S、CFPFP、IJB-C）上有效，还可直接迁移至步态识别（Gait3D，基于骨架关键点），验证了方法对"共享拓扑物体"的通用价值。

## 方法详解
- **背景公式**：标准自注意力 e_ij = Q_i K_j^T / sqrt(d_k)；RPE 扩展为 e'_ij = (Q_i + R^Q_ij)(K_j + R^K_j)^T / sqrt(d_k)。
- **KP-RPE 核心公式**：
  - e'_ij = (Q_i K_j^T + B_ij) / sqrt(d_k)
  - B_ij = f(P)[d(i,j)]，其中 d(i,j) 为 quantized product distance（沿用 iRPE），f(P) 是一个可学习的、由关键点 P 索引的偏置查找表。
- **f(P) 三种实现**：
  1. **Absolute**：f(P) = P' W_L，P' 为 flatten 后的关键点坐标，缺少 query/key 与关键点之间的相对关系。
  2. **Relative**：构建 patch 网格 M ∈ R^{N×2}，计算广播差值 D = Expand(M) − Expand(P)，得 D' ∈ R^{N×(2N_L)}，再 f(P) = D' W_L ∈ R^{N×K}；此时 B_ij 同时依赖 (i,j) 的相对距离和它们各自相对于关键点的位置。
  3. **Multihead Relative**：将 W_L 维度扩展至 H·K（每 attention head 独立），并进一步对每层 transformer block 独立建模，开销几乎不变（f(P) 每前向只算一次）。
- **关键点定义**：人脸为 5 个 facial landmarks（两眼、鼻、两口角）；步态为 17 个 skeleton joint；MNIST toy 为包围盒四角 + 中心。
- **Landmark 检测器**：训练时使用 MobileNetV2-backbone 的 RetinaFace 以加速推理，推理时也使用相同检测器，论文将其命名为 Differentiable Face Aligner。

## 实验与结果
- **数据集**：
  - 训练：MS1MV2、MS1MV3、WebFace4M（112×112，RetinaFace 对齐）；可扩展至 WebFace12M。
  - 低质量：TinyFace、IJB-S（对齐且存在对齐失败）；Unaligned：直接 resize 未做对齐的 CFPFP、IJB-C。
- **Ablation（ViT-small + AdaFace loss，Tab.1/2）**：
  - ViT baseline TinyFace Rank-1: 68.24 → +iRPE: 69.05 → **+KP-RPE: 69.88**；IJB-S: 59.60 → 62.49 → **63.44**；CFPFP unaligned: 72.81 → 77.91 → **93.56**（+15.65 绝对提升）。
  - KP-RPE 变体对比：Relative f(P) 在 unaligned 上从 68.09→89.70，Multihead 进一步 93.56；说明引入 query/key 与关键点相对关系至关重要。
- **SoTA 对比（ViT-Base + AdaFace，Tab.4）**：
  - WebFace4M 训练：ViT+iRPE TinyFace 74.92 → **ViT+KP-RPE 75.80** (+0.88)；IJB-S: 71.93 → **72.78** (+0.85)。
  - WebFace12M 训练：TinyFace 76.18，IJB-S 72.94，CFPFP verification 99.30，IJB-C TAR@0.01% 97.82，全面领先或持平 SOTA。
- **Gait3D（Tab.5，SwinGait-2D baseline）**：Rank-1 67.1→68.2 (+1.1)，mAP 58.76→60.81 (+2.05)。
- **效率（Tab.3）**：ViT-Base +KP-RPE 仅 +0.07 GFLOPs，训练吞吐下降 16.44%；iRPE 同配置 +1.42 GFLOPs，吞吐下降 51.27%。

## 相关工作脉络
1. **RPE / iRPE**（Shaw 2018; Huang 2020; Ramachandran 2019; Wu 2021 iRPE）：本文的定位是在 iRPE 的基础上引入输入dependent 的关键点先验，打破"位置编码与图像内容无关"的静态假设。
2. **AdaFace**（Kim 2022）与 **A-SKD**（Shin 2022）：通过 loss/蒸馏缓解低质量影响，但仍依赖对齐输入；本文从表征层入手，直接让模型容忍未对齐。
3. **MTCNN / RetinaFace / TinaFace**：传统对齐管线中的 landmark 检测器；本文将其轻量级版本作为 KP-RPE 的辅助输入，实现端到端关键点感知的自注意力。
4. **Keypoint-based pooling**（Hachiuma 2023 等）：在 CNN/action 识别中利用关键点做池化；本文首次在 ViT 的 self-attention 中集成关键点，适用于无 pooling 的 Transformer 架构。
5. **低质量人脸识别基准**：TinyFace、IJB-S 等；本文在多个这类 benchmark 上一致验证有效性，并与 ArcFace、CurricularFace、MagFace 等基线对照。
6. **Gait recognition (GaitSet, SwinGait-2D)**：本文展示 KP-RPE 不仅限于人脸，对步态骨架同样有效，体现通用性。

## 局限性与未来方向
- 需要**可用的关键点监督**，对无固定拓扑目标（如通用 object）泛化受限。
- 关键点检测器本身可能出错，尤其极低质量图像的 landmark 噪声会传播至 RPE。
- 论文自述未来方向：**自监督/自发现关键点**，减少对人工标注或独立检测器的依赖。
- 目前仅在 face 和 gait 两类"拓扑一致"任务上验证，尚未扩展到更广泛的视觉任务（如检测、分割）。

## 研究启发与可借鉴点
1. **关键点驱动的相对位置编码范式**可迁移：任何具有稳定拓扑的结构化识别任务（手部识别、姿态估计、器官定位）均可复用此思想。
2. **Relative f(P) 中广播差值网格 D 的设计**值得借鉴——以 O(N·N_L) 的空间复杂度显式编码所有 patch 相对所有 keypoint 的位置关系，计算量小且易嵌入任意 ViT。
3. **以极小参数代价换取对齐鲁棒性**：仅增 0.05M/0.21M 参数，即可在 unaligned 基准上获得巨大提升，适合资源受限部署。
4. **消融设计**：通过"是否使用 query-key 与关键点相对差值"这一变量，清晰区分 Absolute vs Relative f(P) 的收益来源，论证严谨，可作为后续工作的对照范式。
5. **与 Large-scale 数据集可扩展性**：在 WebFace12M 上继续涨点，提示该方法在更大规模训练下仍有增益空间。

## 关键术语表
**KP-RPE**：将关键点位置信息融入 Vision Transformer 相对位置编码的模块，使自注意力偏置随输入图像中关键点动态变化。
**RPE (Relative Position Encoding)**：在 self-attention 中用查询-键相对距离取代绝对坐标的位置编码方式，常用量化欧氏距离或乘积距离。
**iRPE**：引入方向敏感型 quantized x·y product distance 的 RPE 变体，比原始 RPE 更具表达力。
**Differentiable Face Aligner**：论文提出的基于 MobileNetV2-Backbone RetinaFace 的可微 landmark 检测器，推理时用于提供关键点输入 KP-RPE。
**Unaligned dataset**：直接使用原始检测框裁剪图、不做 landmark 对齐的测试集，用于模拟真实场景中的对齐失败。
**Low Quality Aligned dataset**：已尝试对齐但存在大量失败样本的 Benchmark（如 TinyFace、IJB-S），反映现实中的部分对齐误差。
**PartialFC**：大规模 softmax 训练加速技术，用于 WebFace4M 等大身份数据集训练 ViT。
**AdaFace**：面向低质量图像的人脸识别 loss，根据样本质量自适应调整 margin。

## 可复现要素
- **数据集**：MS1MV2、MS1MV3、WebFace4M、WebFace12M（公开）；TinyFace、IJB-S、CFPFP、IJB-C、Gait3D（均有公开协议）；部分低质量数据集需申请。
- **代码/权重**：论文声明"Code and pre-trained models are available"（链接见原文，需访问 author 主页或 GitHub）。
- **超参**：详细设置见 supplementary；主实验使用 AdamW、Cosine LR schedule，ViT-small 做 ablation，ViT-base 做 SOTA 对比；loss 用 AdaFace；训练 augment 含 RandAug、repeated augmentation、random resized crop、blur。
- **Landmark 检测**：主干用 ResNet50-RetinaFace（对齐），KP-RPE 推理用 MobileNetV2-RetinaFace；论文提供了训练 trick 使后者性能接近前者。
