---
title: "Prompt-Driven-Referring-Image-Segmentation-with-Instance-Con"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Shang_Prompt-Driven_Referring_Image_Segmentation_with_Instance_Contrasting_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 15:20:19"
field: "视觉-语言分割"
keywords: ["Referring Image Segmentation", "Prompt Learning", "CLIP", "SAM", "Cross-Modal Interaction", "Instance Contrastive Learning"]
innovations: ["提出Cross-Modal Prompting实现CLIP与SAM端到端双向跨模态交互", "设计Instance Contrastive Learning提升多实例辨别与语言鲁棒性"]
benchmarks: ["RefCOCO", "RefCOCO+", "G-Ref"]
---

# 论文速读：Prompt-Driven-Referring-Image-Segmentation-with-Instance-Con

## 一句话总结
本文提出了Prompt-RIS框架，通过prompt learning将CLIP和SAM端到端桥接，利用Cross-Modal Prompting实现双向跨模态交互以适配像素级任务，并结合Instance Contrastive Learning提升模型对多实例场景的辨别能力，在Referring Image Segmentation任务上取得SOTA性能。

## 研究问题与动机
- **现有CLIP适配方法不足**：大多数将CLIP用于分割任务的方法采用两阶段"mask-classification"模式（先生成mask proposals再分类），缺乏足够的跨模态交互，难以捕捉文本-pixel对应关系。
- **单向prompt learning局限性**：现有prompt learning方法多为单向信息流（vision-to-language或language-to-vision），无法充分进行cross-modal交互，对像素级RIS任务不适用。
- **SAM直接使用存在困难**：SAM在RIS任务中缺少point prompts时难以生成精确分割结果；若将mask-cropped图像输入CLIP生成近似text prompts，会忽略referent的全局上下文信息。
- **多实例、多描述的挑战**：同一图像常包含多个被描述实例，同一实例又有多种不同描述方式，模型需具备强判别力和对语言多样性的鲁棒性。

## 核心贡献（创新点）
- **结构创新**：提出Prompt-RIS框架，通过prompt learning将CLIP和SAM端到端桥接，将两者丰富知识和强大能力迁移至RIS任务。与两阶段mask-classification方法的本质区别在于避免了信息丢失和交互不足。
- **方法创新1**：提出Cross-Modal Prompting (CMP)，在CLIP编码过程中实现双向信息流提示学习（V2L和L2V），使视觉和语言模态通过prompt互相生成上下文信息，实现更充分的跨模态交互和fine-grained text-pixel对齐。
- **方法创新2**：提出Instance Contrastive Learning (ICL)，通过对比学习使描述相同实例的不同句子的预测mask相似，同时抑制描述不同实例的mask重叠，提升模型对多实例的辨别力。
- **多prompt生成机制**：CLIP生成的coarse mask、selected point和projected text features作为三类prompt输入SAM，实现更准确的分割预测。

## 方法详解
- **整体架构**：Prompt-RIS包含CLIP（ViT-B/16）和SAM（ViT-B/16）两部分，图像分别resize到480×480（CLIP）和1024×1024（SAM）。
- **Image&Text Encoding**：图像patches投影为embedding后输入CLIP编码器；文本添加[SOS]/[EOS]后输入CLIP文本编码器。中间层特征通过CMP进行跨模态交互。
- **Cross-Modal Prompting (CMP)**：在CLIP每层编码中，引入可学习token $T_V^i$（数量n=16）作为query，通过cross-attention从视觉特征中提取V2L prompt；对称地，L2V prompt从文本特征生成。公式：$T_{V2L}^i = CrossAttn(T_V^i, F_{I_C}^i)$，$T_{L2V}^i = CrossAttn(T_L^i, F_T^i)$。
- **CLIP-based Prompting**：经cross-attention更新后，取$F_T'$中[EOS]位置特征作为全局文本特征$t_q$，计算text-pixel响应图$S_C' = F_{I_C}' \cdot t_g$，reshape后得到低分辨率coarse mask。
- **Mask/Point/Text Prompt生成**：
  - Mask prompt：$S_C'$上采样后通过SAM prompt encoder映射。
  - Point prompt：对$S_C$做Gumbel-Softmax（straight-through estimator）得到可微的one-hot point map $M$，与预计算的SAM位置embedding加权求和：$e_p = \sum M_i E_i$。
  - Text prompt：使用self- and cross-attention transformer blocks，以$F_T$为query，投影到SAM图像特征上。
- **SAM Decoding**：$S' = D_{sam}([T_{iou}; T_{out}; P_{point}; P_{text}], (P_{mask} + F_{I_S}))$，再上采样4倍得最终预测。
- **Instance Contrastive Learning**：对同一图像采样b=4个句子，计算mask预测间的Dice系数重叠$O_{ij}$，损失函数：$L_{icl} = \frac{1}{b^2}\sum\sum -w_i(Y_{ij}\log(O_{ij}) + (1-Y_{ij})\log(1-O_{ij}))$，其中$w_i$为mask与ground truth的IoU，防止错误预测误导。
- **总损失**：$L = L_{clip\_seg} + L_{sam\_seg} + L_{icl}$，采用BCE+Dice loss。

## 实验与结果
- **数据集**：RefCOCO、RefCOCO+、G-Ref（均基于MS-COCO），以及open-vocabulary setting下的seen/unseen类别划分。
- **评估指标**：oIoU、mIoU、P@X（X∈{0.5, 0.7, 0.9}）。
- **主要结果**：
  - RefCOCO val: oIoU 76.36 / mIoU 78.10（vs. CGFormer 74.75/76.93）
  - RefCOCO+ val: oIoU 67.06 / mIoU 71.13（vs. CGFormer 64.54/68.56）
  - G-Ref val(G): oIoU 64.79 / mIoU 69.17（vs. CGFormer 62.51/65.79）
  - 相比SOTA平均提升：RefCOCO oIoU +2.11, mIoU +1.67；RefCOCO+ oIoU/mIoU均+2.5；G-Ref oIoU +2.89, mIoU +3.25。
- **泛化能力**：在unseen类别上，相比CGFormer平均提升超2.5 points（mIoU）。
- **Ablation**：CMP带来mIoU +5.16；SAM带来mIoU +4.3；ICL在P@0.9上提升3.32 points。

## 相关工作脉络
- **CLIP-based RIS方法（CRIS、ETRIS）**：采用cross-modal decoder适配CLIP至像素级任务，但交互不足；本文通过CMP实现双向prompt而非decoder，保留CLIP预训练知识。
- **Prompt Learning方法（CoOp、Co-CoOp、MaPLe）**：多为image-level任务的单向prompt；本文扩展到pixel-level并设计双向prompt。
- **RIS SOTA方法（CGFormer、LAVT、DMMI）**：基于Transformer的cross-modal fusion；本文引入SAM生成精细mask，结合prompt learning实现更强泛化。
- **Mask-generation方法（Maskformer、SAM）**：传统两阶段mask-classification忽略全局上下文；本文端到端桥接，CLIP直接生成SAM所需的多种prompt。
- **Vision-Language Transformer（VLT、ReSTR）**：基于attention的多模态融合；本文侧重prompt-driven的大模型适配。

## 局限性与未来方向
- **计算效率**：双encoder（CLIP+SAM）及多层CMP带来较高计算开销，推理速度可能受限。
- **训练策略**：两阶段训练（先训CLIP+CMP 20epoch，再端到端30epoch）可能影响收敛稳定性。
- **prompt数量依赖**：CMP中learnable token数量n需调优，未探索不同场景的最优设置。
- **潜在方向**：可探索更轻量的prompt生成机制、单阶段端到端训练、或扩展至video/3D referring segmentation。

## 研究启发与可借鉴点
- **双向prompt设计**：CMP的V2L+L2V对称结构可有效增强跨模态交互，可迁移至其他vision-language下游任务。
- **Gumbel-Softmax可微点选择**：用straight-through estimator实现mask-to-point的可微映射，解决预训练模型冻结时的联合训练难题，值得借鉴。
- **Instance-level对比学习**：ICL利用同一图像的多句子监督信号，提升多实例辨别力，可应用于多对象定位/分割任务。
- **多类型prompt融合**：mask+point+text三种prompt协同输入SAM的策略，为利用大模型能力提供了通用范式。
- **Open-vocabulary泛化验证**：seen/unseen类别划分评估方式值得参考，可用于验证模型的知识迁移能力。

## 关键术语表
- **Referring Image Segmentation (RIS)**：给定自然语言描述，在图像中分割出对应目标区域的视觉-语言任务。
- **Cross-Modal Prompting (CMP)**：在CLIP编码层中实现vision-to-language和language-to-vision双向prompt生成，增强跨模态交互。
- **Instance Contrastive Learning (ICL)**：通过对同一图像的不同描述句子生成mask进行对比学习，提升实例辨别力。
- **Gumbel-Softmax**：可微分的类别采样技巧，本文用于将coarse mask转换为point prompt的位置embedding加权。
- **Open-vocabulary Setting**：评估模型在未见过类别上的泛化能力，将数据集分为seen和unseen子集测试。
- **Prompt Learning**：冻结预训练模型参数，仅学习额外prompt tokens以适配下游任务的技术。
- **Text-Pixel Alignment**：将文本中不同词汇/短语与图像中对应像素区域进行精确匹配的能力。
- **Mask-Classification Paradigm**：两阶段分割方法，先生成class-agnostic mask proposals再分类前景。

## 可复现要素
- **数据集**：RefCOCO、RefCOCO+、G-Ref（基于MS-COCO），均公开可用。
- **代码/权重**：论文未明确提及开源，需关注作者后续发布。
- **关键超参**：ViT-B/16 backbone；CLIP输入480×480，SAM输入1024×1024；CMP token数n=16；batch size=64（16 image × 4 sentences）；学习率1e-4，AdamW，多项式衰减power=0.9；训练50 epoch（前20单阶段，后30端到端）；prompt dropout策略。
