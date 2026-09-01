---
title: "Parameter-Efficient-Fine-tuning-via-Cross-Block-Orchestratio"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Peng_Parameter_Efficient_Fine-tuning_via_Cross_Block_Orchestration_for_Segment_Anything_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 21:30:56"
field: "视觉基础模型高效微调"
keywords: ["PEFT", "SAM", "cross-block orchestration", "hyper-complex number", "medical image segmentation"]
innovations: ["提出跨块通信(IBC)模块，通过可学习关系矩阵突破HMC限制实现参数空间全局协同", "设计超复数层(HL)增强块内投影方向调整对全局参数空间的影响力", "双系数集设计兼顾局部信息传递与跨层全局优化"]
benchmarks: ["COCO", "TRCAN", "NWPU", "SSDD", "SONAR", "ADOME", "SPLEN", "MOMO", "BRAST", "SEGRAP"]
---

# 论文速读：Parameter-Efficient-Fine-tuning-via-Cross-Block-Orchestration

## 一句话总结
本文提出 SAM-COBOT，通过在 PEFT 模块中引入跨块通信（IBC）和块内增强（IBE）机制，突破隐马尔可夫链对参数空间的限制，以仅约 1K 额外参数的代价显著提升 SAM 在医疗、遥感与自然图像分割任务上的性能。

## 研究问题与动机
- **PEFT 在分割任务中的局限性**：现有 PEFT 方法（如 LoRA、Adaptformer）主要面向图像分类设计，而分割任务输出空间更大、更复杂，需要更强的参数投影方向调整能力。
- **隐马尔可夫链（HMC）的约束**：传统 PEFT 在各层独立注入少量参数，受限于 HMC 的"无记忆性"，各层状态仅受相邻层影响，导致对参数空间的调整幅度有限。
- **跨块协同缺失**：深层网络不同 block 之间存在互补学习模式（浅层保留细节、深层捕获语义），但现有方法未能在参数空间层面实现跨 block 协同。
- **超复数结构的引入空白**：超复数层（hyper-complex layer）在促进正交投影方向通信方面的潜力尚未在 PEFT 中被探索。

## 核心贡献（创新点）
1. **提出 SAM-COBOT 框架**：首次将跨块编排（cross-block orchestration）引入 SAM 的 PEFT，使参数空间的投影方向能够在整个模型中进行协调调整，区别于传统逐层独立微调。
2. **设计跨块通信模块（IBC）**：引入可学习关系矩阵 S，将各 PEFT 块的系数集张量化，通过 T-product 和张量变换实现跨层信息交互，突破 HMC 限制。
3. **设计块内增强模块（IBE）**：利用超复数层（HL）生成线性投影头权重，通过 Hamilton 积在超球面上实现旋转变换，增强单次投影方向调整对全局参数空间的影响力。
4. **双系数集设计**：保留一个在 HMC 约束下通信的系数集以维持任务相关信息的传播，同时引入另一个跨层通信的系数集以实现全局调整，兼顾局部与全局。
5. **轻量级且通用的插件式方案**：仅需约 1K 额外参数即可无缝集成到 LoRA 或 Adaptformer 等现有 PEFT 方法中，在 10 个数据集上均实现显著提升。

## 方法详解
**整体架构**：SAM-COBOT 在每个 PEFT 块中依次接入 IBC 和 IBE 模块，不修改 SAM 原始结构，所有预训练权重保持冻结。

**跨块通信模块（IBC）**：
- 将每个 PEFT 块的参数空间分解为基集合（base set）和系数集合（coefficient set），因基参数远多于系数，选择系数集作为通信媒介。
- 引入可学习对角矩阵 $\Lambda$，其对角元素构成系数集 $\{\lambda_i\}$。
- 将所有 $L$ 个 transformer block 的系数集张量化为 $\mathcal{T} = [\Lambda_1, \Lambda_2, \cdots, \Lambda_L] \in \mathbb{R}^{V \times V \times L}$。
- 为避免梯度传播中的跨切片信息损失，采用 T-product（张量积）定义跨切片交互，并通过可学习可逆关系矩阵 $\mathbf{S} \in \mathbb{R}^{L \times L}$ 实现张量变换：$\mathcal{T}_w = \mathcal{T} \times_3 \mathbf{S}$，其中 $\times_3$ 表示 mode-3 乘积。
- 双系数集设计：$\{\Lambda_\ell^{\mathrm{MC}}\}$ 在 HMC 约束下通信以保留层间信息传递优势；$\{\Lambda_\ell^{\mathrm{LM}}\}$ 与其他 block 的系数集进行跨层通信。

**块内增强模块（IBE）**：
- 引入超复数层（HL），其权重由超球面上的元素经 Hamilton 积生成。
- 4 维超复数形式：$h = a_0 \cdot 1 + a_1 j_1 + a_2 j_2 + a_3 j_3$，满足 $j_1^2 = j_2^2 = j_3^2 = j_1 j_2 j_3 = -1$，且乘法非交换。
- Hamilton 积实现对超球面上元素的旋转与缩放变换：$H = \widetilde{H_a} \otimes \widetilde{H_b}$。
- 通过实变换 $\mathbb{R}^N \to \mathbb{R}^V$ 将超复数空间映射回参数空间，得到投影头权重 $\mathbf{W}$。
- 该设计使各系数元素的调整能够产生更大的投影方向影响。

**输出与损失**：
- 第 $\ell$ 个模块的输出特征：$\widetilde{\mathbf{M}}_\ell = \mathcal{F}_\ell(\mathbf{M}_\ell; \mathbf{W}\Lambda^{\mathrm{MC}}) + \mathcal{F}_\ell(\mathbf{M}_\ell; \mathbf{W}\Lambda^{\mathrm{LM}})$。
- 损失函数：$\mathcal{L} = \mathcal{L}_{\mathrm{ce}} + \mathcal{L}_{\mathrm{dice}}$（二元交叉熵 + Dice 损失）。

## 实验与结果
**数据集**：10 个数据集覆盖三大领域——
- 自然图像：COCO、TRCAN
- 遥感图像：NWPU、SSDD、SONAR
- 医学图像：ADOME、SPLEN、MOMO、BRAST、SEGRAP

**评估指标**：DSC（医学分割）、mIoU（自然/遥感分割）

**基线**：LoRA、Adaptformer（均集成 SAM-COBOT 后对比）

**核心结果**：
| 方法 | 参数量(K) | 提升幅度（典型） |
|------|-----------|------------------|
| LoRA | 147.4 | Baseline |
| LoRA+Ours | 148.3 (+0.9) | COCO +0.3%, ADOME +0.9%, BRAST +0.5% |
| Adaptformer | 322.7 | Baseline |
| Adaptformer+Ours | 324.0 (+1.3) | COCO +0.5%, ADOME +1.2%, SEGRAP +1.0% |

**关键结论**：
- SAM-COBOT 在医学分割上提升最为显著，ADOME 数据集较 Adaptformer 提升 **1.2%** DSC，SEGRAP 提升 **1.0%**。
- 仅增加约 **1K** 参数即可实现稳定提升，在低维隐藏空间（$V \leq 16$）时增益更加明显（ADOME 提升超过 **1.6%**）。
- 在 ViT-Base 和 ViT-Large 两种 backbone 上均有效，证明方法具有良好的架构通用性。
- 消融实验验证：关系矩阵（RM）贡献最大（ADOME 提升 0.8%），超复数层（HL）次之（0.4%）。

## 相关工作脉络
1. **LoRA**：通过低秩分解矩阵 $\omega = \beta\alpha$ 替代额外模块，仅调整投影方向但受限于单块独立优化，缺乏跨层协同。
2. **Adaptformer**：在每层 MLP 旁路引入下投影-上投影分支，参数量较大（322.7K），但同样受 HMC 约束。
3. **LST（Ladder Side-Tuning）**：引入可学习侧适配器，但层间更新仍受相邻层限制，无法突破 HMC。
4. **Visual Prompt Tuning (VPT)**：在输入端添加可学习 prompt，与 PEFT 的正则化思路不同，不直接调整参数空间投影方向。
5. **Cross-block 相关工作**（CLR-Net、SegFormer 等）：在特征空间进行跨层交互，而本文首次在**参数空间**层面实现跨 block 编排。
6. **超复数神经网络**（Quaternet 等）：利用四元数/超复数表示旋转对称性，本文首次将其引入 PEFT 的参数空间增强。

## 局限性与未来方向
- **关系矩阵 S 的计算开销**：虽仅增加约 1K 参数，但 T-product 涉及的张量变换在训练时可能带来一定计算负担，未详细讨论推理延迟。
- **仅验证了 SAM 一个基础模型**：虽测试了 ViT-Base 和 ViT-Large，但未涉及更大型的 ViT-Huge 版本。
- **固定超复数维度**：论文固定 $n=4$（四元数维度），未探索不同维度对性能的影响。
- **仅应用于图像分割**：方法的有效性尚未在视频分割、3D 分割等其他视觉任务中验证。
- **未来方向**：可扩展至其他foundation模型（如 foundation model for video/3D）、探索更高效的张量分解方式以降低计算开销、研究动态关系矩阵（自适应块间通信强度）。

## 研究启发与可借鉴点
1. **参数空间层面的跨块通信思想**：将 HMC 限制转化为参数空间的张量交互问题，为其他 PEFT 方法（如针对 NLP 的 cross-layer tuning）提供了新思路。
2. **超复数层的正交性保障**：通过 Hamilton 积在超球面上进行旋转，天然保持权重正交性，可迁移至任何需要保持参数空间几何结构的微调场景。
3. **双系数集设计策略**：在"保留原有信息传递路径"与"引入全局协同"之间取得平衡，适用于任何深度网络的参数高效微调。
4. **低维场景下的增效机制**：在隐藏维度 $V \leq 16$ 时增益最显著，提示后续研究可在极低参数预算下优先采用此类跨块通信方案。
5. **实验设计借鉴**：跨三大视觉领域（自然/遥感/医学）的系统性验证，以及不同 backbone 尺寸、不同隐藏维度的多维消融，为后续工作提供了完整的评估范式。

## 关键术语表
- **PEFT（Parameter-Efficient Fine-tuning）**：参数高效微调，仅微调少量参数而冻结主体模型，以降低大模型适配成本。
- **SAM（Segment Anything Model）**：Meta 提出的视觉基础分割模型，包含 ViT 图像编码器和轻量 mask 解码器。
- **HMC（Hidden Markov Chain）**：隐马尔可夫链，此处指深度网络中各层状态仅受相邻层影响的无记忆传播约束。
- **T-product**：张量积，用于三维张量间运算的特殊乘积定义，可实现跨切片信息交互。
- **Hyper-complex Number**：超复数，扩展复数的代数结构，本文使用 4 维（含实部与三个虚部）。
- **Hamilton Product**：Hamilton 积，超复数之间的乘法运算，保持非交换性，对应超球面上的旋转变换。
- **Relation Matrix (S)**：可学习的 $L \times L$ 关系矩阵，用于捕捉不同 PEFT block 系数集之间的依赖关系。
- **Dual Coefficient Sets**：双系数集，分别负责 HMC 约束下的层间通信和跨 block 全局通信。

## 可复现要素
- **数据集**：COCO、TRCAN、NWPU、SSDD、SONAR、ADOME、SPLEN、MOMO、BRAST、SEGRAP（均为公开数据集）
- **代码/权重**：论文未提及代码开源状态，SAM 官方权重可从 HuggingFace 获取
- **关键超参**：
  - 训练 epoch：医学 25 / 自然与遥感 20
  - 学习率：医学 $1.25 \times 10^{-6}$ / 其他 $10^{-4}$
  - Weight decay：医学 $5 \times 10^{-4}$ / 其他 $5 \times 10^{-5}$
  - Backbone：ViT-Base（SAM 默认）
  - Prompt：box prompt + 随机扰动 0-50 像素
  - 优化器：Adam
