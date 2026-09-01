---
title: "KeyPoint-Relative-Position-Encoding-for-Face-Recognition"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Kim_KeyPoint_Relative_Position_Encoding_for_Face_Recognition_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 20:14:27"
field: "人脸/步态识别"
keywords: ["face recognition", "vision transformer", "relative position encoding", "affine robustness", "keypoint", "gait recognition", "low-quality images"]
innovations: ["提出KP-RPE将ViT相对位置编码动态依赖关键点，提升仿射变换鲁棒性", "设计Relative/MultiHead f(P)变体，证明相对坐标比绝对坐标更有效", "在低质量与未对齐数据集上显著提升性能且计算开销远低于iRPE"]
benchmarks: ["TinyFace", "IJB-S", "CFPFP", "IJB-C", "Gait3D"]
---

# 论文速读：KeyPoint-Relative-Position-Encoding-for-Face-Recognition

## 一句话总结
本文提出 **KP-RPE（KeyPoint Relative Position Encoding）**，将 ViT 中的相对位置编码（RPE）与图像关键点（如面部 landmarks）相结合，使自注意力机制中的 query-key 关系动态依赖于关键点位置，从而显著提升模型对未见过仿射变换（缩放、平移、旋转）的鲁棒性，在低质量/未配准人脸及步态识别任务上均取得显著提升。

## 研究问题与动机
- **ViT 对未见仿射变换鲁棒性不足**：原始 Absolute Position Embedding（Abs-PE）在 MNIST→AffNIST 等跨仿射测试中性能从 98.1% 骤降至 77.27%，而 RPE 有所改善但仍存在局限。
- **传统 RPE 的静态性缺陷**：RPE（含 iRPE）仅以像素间相对距离计算注意力偏置，无法感知语义关键点，导致缩放/旋转后 query-key 语义关系变化但偏置不变。
- **人脸等识别任务依赖精确配准**：现有 SoTA 模型（如 ArcFace、AdaFace）均训练于对齐数据，低质量图像（模糊、暗光）中 landmark 检测易失败，导致测试时配准错误。
- **现有低质量识别方法未解决配准失效问题**：AdaFace、A-SKD 等通过损失设计或蒸馏缓解质量问题，但未从根本上增强模型对几何变换的泛化能力。

## 核心贡献（创新点）
- **洞察 RPE 对仿射鲁棒性的提升价值**：首次系统验证 RPE/iRPE 可使 ViT 在不重新训练的情况下对未见仿射变换具有一定鲁棒性，为后续改进奠定基础。
- **提出 KP-RPE，将 RPE 偏置动态化为关键点的函数**：令注意力偏移 $\mathbf{B}_{ij} = f(\mathbf{P})[d(i,j)]$ 依赖于图像关键点 $\mathbf{P}$，使 query-key 关系随关键点位置自适应调整，区别于 iRPE 的静态距离查表。
- **设计三种 f(P) 变体并证明相对形式必要性**：Absolute f(P) 仅用关键点绝对坐标效果差（IJB-S Rank-1 仅 9.97%），Relative f(P) 引入 query/key 相对于关键点的偏移后大幅提升，MultiHead f(P) 进一步分头独立建模带来增益。
- **计算效率显著优于 iRPE**：KP-RPE 仅增加 0.02/0.07 GFLOP（ViT-S/B），训练吞吐下降 9.15%/16.44%，而 iRPE 分别增加 0.71/1.42 GFLOP 并导致吞吐下降 44%/51%。
- **跨任务泛化验证**：不仅提升人脸（TinyFace、IJB-S、CFPFP、IJB-C）识别，在步态识别（Gait3D）上同样有效，证明方法的通用性。

## 方法详解
- **基础 RPE 公式**：在 Self-Attention 中，注意力得分修正为 $e'_{ij} = \frac{\mathbf{Q}_i \mathbf{K}_j^T + \mathbf{B}_{ij}}{\sqrt{d_k}}$，其中 $\mathbf{B}_{ij} = \mathbf{W}[d(i,j)]$ 是基于距离索引的可学习偏置表。
- **KP-RPE 核心设计**：保持距离函数 $d(i,j)$ 不变（以保证 distance caching 效率），转而让偏置表本身 $\mathbf{W}$ 依赖关键点 $\mathbf{P}$，即 $\mathbf{B}_{ij} = f(\mathbf{P})[d(i,j)]$。
- **Relative f(P) 推导**：构建 patch 网格 $\mathbf{M} \in \mathbb{R}^{N \times 2}$，计算与关键点 $\mathbf{P}$ 的广播差值 $\mathbf{D} = \text{Expand}(\mathbf{M}) - \text{Expand}(\mathbf{P}) \in \mathbb{R}^{N \times N_L \times 2}$，经线性投影得到 $f(\mathbf{P}) = \mathbf{D}'\mathbf{W}_L \in \mathbb{R}^{N \times K}$，最终 $\mathbf{B}_{ij} = f(\mathbf{P})[i, d(i,j)]$。
- **MultiHead f(P)**：将 $\mathbf{W}_L$ 扩展至 $\mathbb{R}^{(2N_L) \times N_d \cdot HK}$，为每个 transformer 层和每个 attention head 独立建模 keypoint-adaptive 偏置，几乎不增加额外计算开销。
- **关键点来源**：人脸使用 5 个 landmarks（两眼、鼻尖、两嘴角），步态使用 17 个关节点；检测器采用轻量级 MobileNetV2-Backbone RetinaFace，保证推理效率。

## 实验与结果
- **数据集**：训练集 MS1MV2/MS1MV3/WebFace4M/WebFace12M；测试集分三类——高质量对齐（CFPFP、IJB-C）、低质量/易配准失败（TinyFace、IJB-S）、故意未对齐（Raw CFPFP、Raw IJB-C）；步态使用 Gait3D。
- **Ablation（Tab.1）**：ViT+KP-RPE 在 TinyFace Rank-1 达 69.88%（+1.64 vs iRPE），IJB-S Rank-1 达 63.44%（+0.95）；未对齐 CFPFP Verification 达 93.56%（vs ViT 72.81%，+20.75）；未对齐 IJB-C TAR@0.01% 达 91.85%（vs ViT 21.62%，+70.23）。
- **f(P) 变体对比（Tab.2）**：Absolute f(P) 在未对齐 IJB-S 仅 9.97%，Relative f(P) 恢复至 62.51%，MultiHead 进一步提升至 63.44%，验证相对坐标设计的必要性。
- **SoTA 对比（Tab.4，WebFace4M）**：ViT+KP-RPE 在 TinyFace Rank-1 达 75.80%（+0.88 vs iRPE），IJB-S Rank-1 达 72.78%（+0.85），高质量数据集保持竞争力（AgeDB 97.67%、CFPFP 99.01%、IJB-C 97.13%）。
- **可扩展性（WebFace12M）**：ViT+KP-RPE 在 TinyFace 达 76.18%，IJB-S 达 72.94%，CFPFP 99.30%，IJB-C 97.82%，持续领先。
- **步态识别（Tab.5，Gait3D）**：SwinGait-2D + KP-RPE Rank-1 达 68.2%（+1.1）、mAP 60.81（+2.05），验证跨任务泛化。

## 相关工作脉络
- **RPE 基础工作**：Shaw et al. [60] 首次将 RPE 引入 Transformer；Huang et al. [28] 改进为 Better RPE；Ramachandran et al. [53] 提出 2D 分离式 RPE。本文定位：将这些 RPE 变体应用于仿射鲁棒性分析，并进一步扩展到关键点依赖版本。
- **iRPE [74]**：引入方向感知的乘积距离和 Q/K 交互，是本文基线 RPE。本文区别：iRPE 偏置仅依赖固定网格距离，KP-RPE 使偏置表随输入关键点动态变化。
- **低质量人脸识别**：AdaFace [30] 提出质量自适应 margin loss；A-SKD [63] 用教师蒸馏聚焦相似区域。本文区别：这些方法仍训练于对齐数据，KP-RPE 从特征编码层面增强几何鲁棒性，直接受益于配准错误场景。
- **关键点检测与配准**：MTCNN [81]、RetinaFace [10]、TinaFace [90] 是主流人脸 landmark 检测器。本文区别：此前工作将 keypoints 用于显式几何对齐，KP-RPE 将其作为 RPE 的动态条件注入 Transformer 内部，无需显式重映射。
- **Keypoint-based Pooling**：[23] 提出基于关键点的池化机制用于动作识别。本文区别：KP-RPE 直接修改 self-attention 偏置，无需 pooling 结构，可直接插入任意 ViT/SwinTransformer。
- **步态识别相关工作**：GaitSet [6]、SwinGait-2D [17]、DyGait [71] 等。本文定位：首次将关键点辅助的相对位置编码引入步态识别，利用 2D 骨架关键点提升 gait feature 建模。

## 局限性与未来方向
- **依赖预定义/检测的关键点**：方法要求图像具有稳定拓扑结构（人脸、人体），对无固定形态的对象（如动物、物体）难以直接应用。
- **关键点检测误差的传递**：尽管论文论证了少量噪声关键点比全局配准错误影响更小，但极端情况下检测失败仍会影响性能。
- **未来方向**：论文建议探索**自学习关键点**（self-discovery of keypoints），减少对额外检测器的依赖，提升在非标准拓扑数据上的通用性。

## 研究启发与可借鉴点
- **RPE 作为仿射鲁棒性的基础构件**：可将 RPE/iRPE 作为默认选择替换 Abs-PE，在不增加显著计算开销的前提下获得基础几何泛化能力，适合所有 ViT-based 视觉任务。
- **动态偏置表设计范式**：将注意力偏置设计为输入的函数（而非固定查表）是一条有效路径，可迁移至目标检测、分割等需要空间感知的任务。
- ** Relative 坐标比 Absolute 更关键**：Ablation 显示 Absolute f(P) 几乎无效（IJB-S 仅 9.97%），必须引入 query/key 相对于关键点的偏移，这一设计原则对其它关键点辅助模块有参考价值。
- **高效性与性能兼顾**：KP-RPE 通过"仅让偏置表依赖 P、距离函数保持静态"的设计实现了 distance caching，这对部署敏感场景具有借鉴意义。
- **与 AdaFace 等质量感知方法的正交性**：KP-RPE 可从编码层面增强鲁棒性，与 AdaFace 的损失设计、A-SKD 的蒸馏策略互补，可结合使用。

## 关键术语表
- **KP-RPE**：KeyPoint Relative Position Encoding，将 ViT 中 RPE 的注意力偏置动态依赖于图像关键点的新型位置编码方法。
- **Relative Position Encoding (RPE)**：不依赖绝对坐标、仅编码 query-key 相对距离的位置编码方式，使模型具备平移不变性。
- **iRPE**：Improved RPE，引入方向感知乘积距离和 Q/K 交互的 RPE 改进版本，本文的主要 RPE 基线。
- **Affine Transformation**：仿射变换，包括缩放、旋转、平移及组合，本文关注的测试时几何扰动类型。
- **Facial Landmarks**：面部关键点（通常为 5 个：两眼、鼻尖、两嘴角），作为 KP-RPE 中关键点 P 的定义来源。
- **Self-Attention**：Transformer 核心机制，通过 query-key 相似度计算注意力权重，KP-RPE 在此机制中注入关键点依赖的偏置。
- **AdaFace**：Kim et al. 提出的质量自适应 margin loss，将图像质量估计融入识别训练，本文在其基础上叠加 KP-RPE。
- **Gait Recognition**：步态识别，通过分析人体行走模式进行身份识别，本文验证 KP-RPE 在非人脸拓扑任务上的通用性。

## 可复现要素
- **代码/权重**：论文声明 "Code and pre-trained models are available"（论文未提供具体链接，需查看原文附录或作者主页）。
- **训练数据集**：MS1MV2、MS1MV3、WebFace4M、WebFace12M（部分公开，部分需申请）。
- **测试数据集**：TinyFace [7]、IJB-S [29]、CFPFP [59]、IJB-C [48]、Gait3D [87]（均为公开数据集）。
- **关键点检测器**：RetinaFace，backbone 为 MobileNetV2（推理用）和 ResNet50（训练/对齐用）。
- **关键超参**：学习率、weight decay 等见 supplementary material；AdamW 优化器；Cosine LR scheduler；ViT-Small 用于 ablation，ViT-Base 用于 SoTA 对比；AdaFace loss；PartialFC 用于 WebFace4M 大分类器训练。
- **输入分辨率**：112×112。
