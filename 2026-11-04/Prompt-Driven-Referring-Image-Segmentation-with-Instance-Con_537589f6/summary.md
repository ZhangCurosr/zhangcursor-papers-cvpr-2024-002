---
title: "Prompt-Driven-Referring-Image-Segmentation-with-Instance-Con"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Shang_Prompt-Driven_Referring_Image_Segmentation_with_Instance_Contrasting_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 00:45:06"
field: "视觉语言理解"
keywords: ["referring image segmentation", "prompt learning", "CLIP", "SAM", "cross-modal prompting", "instance contrastive learning"]
innovations: ["提出双向Cross-Modal Prompting适配CLIP至像素级RIS任务", "设计Instance Contrastive Learning增强多实例判别与多描述鲁棒性", "通过prompt learning端到端桥接CLIP与SAM，生成mask/point/text三类prompt"]
benchmarks: ["RefCOCO", "RefCOCO+", "G-Ref"]
---

# 论文速读：Prompt-Driven-Referring-Image-Segmentation-with-Instance-Con

## 一句话总结
论文提出Prompt-RIS框架，通过prompt learning将CLIP与SAM端到端连接，利用Cross-Modal Prompting（CMP）和Instance Contrastive Learning（ICL）解决RIS任务中细粒度图文对齐与多实例区分难题，在RefCOCO、RefCOCO+、G-Ref三个基准上均达到SOTA。

## 研究问题与动机
1. **CLIP缺乏像素级能力**：CLIP是图像级预训练模型，直接用于RIS等细粒度分割任务效果不佳，现有方法多采用"mask-classification"两阶段范式，忽略了文本与像素间的细粒度对应关系。
2. **单向提示无法充分交互**：现有prompt learning方法多为单向（vision-to-language或language-to-vision），而RIS需要充分的跨模态交互来实现文本内容与图像区域的一一匹配。
3. **忽略全局上下文与实例关系**：RIS描述往往包含位置、属性、实例间关系，而非显式类别，现有方法仅关注mask内前景实例，忽略了图像中多实例间的上下文关系。
4. **SAM在无point prompt时精度受限**：直接使用SAM进行RIS需要额外的人为点提示，且基于mask裁剪再输入CLIP的方式也忽略了referent的全局上下文。

## 核心贡献（创新点）
1. **提出Prompt-RIS框架**：首次通过prompt learning将CLIP和SAM端到端桥接，将两者的知识和能力迁移到RIS任务，区别于以往"mask+分类"的两阶段方案。
2. **Cross-Modal Prompting（CMP）方法**：在CLIP编码的各中间层引入双向（V2L和L2V）prompt learning机制，实现细粒度的图文交互和对齐，区别于CoOp/MaPLe等单向提示方法。
3. **Instance Contrastive Learning（ICL）**：在同一张图像的多个描述之间引入对比学习，提升模型对不同实例的判别力和对同一实例多种表达的鲁棒性，这是本文针对RIS多实例特性的专门设计。
4. **多类型prompt生成机制**：CLIP生成的特征可同时作为SAM的mask prompt、point prompt和text prompt，实现了三个prompt模态的统一生成与端到端训练。

## 方法详解
- **整体架构**：CLIP负责图文编码与交叉提示，生成粗粒度mask；SAM负责精细分割，接收CLIP提供的三类prompt（mask、point、text）。
- **Cross-Modal Prompting（CMP）**：在CLIP ViT编码的各层中，引入可学习token（数量n=16），通过cross-attention分别与图像patch特征或文本token交互，实现V2L（图像→文本）和L2V（文本→图像）双向提示，促进图文细粒度对齐。
- **CLIP-based Prompting**：编码后通过小型cross-attention decoder更新特征，用[eos]位置文本特征与图像特征内积得到粗mask；该mask上采样后经SAM prompt encoder转为mask prompt；通过Gumbel-Softmax可微地采样point坐标并查找预计算的position embedding；文本特征经self/cross-attention转为SAM的text prompt。
- **SAM Decoding**：使用mask prompt + point prompt + text prompt三类提示输入冻结的SAM mask decoder，生成高分辨率最终mask。
- **Instance Contrastive Learning（ICL）**：对同一图像采样b条描述，计算b个mask间的Dice重叠矩阵O，构造二元标签Y（同实例为正），结合IoU加权构建对比损失$L_{icl}$，增强实例间区分与实例内一致。
- **损失函数**：$L = L_{clip\_seg} + L_{sam\_seg} + L_{icl}$，其中分割损失为粗mask和最终mask分别计算的BCE+Dice。

## 实验与结果
- **数据集**：RefCOCO、RefCOCO+、G-Ref（均基于MS-COCO）。
- **评估指标**：oIoU、mIoU、P@X（X∈{0.5, 0.7, 0.9}）。
- **主要结果（oIoU/mIoU）**：
  - RefCOCO val：76.36 / 78.10，较CGFormer（74.75/76.93）分别提升1.61/1.17；testA：80.37/81.21，testB：72.29/74.64。
  - RefCOCO+ testA：73.58/76.60，testB：58.96/64.25，较CGFormer分别提升约2.5+点。
  - G-Ref val(G)：64.79/69.17，val(U)：67.16/70.47，test(U)：69.01/71.29，较CGFormer分别提升2.28/3.25。
- **泛化能力（unseen categories，mIoU）**：在三个数据集的unseen split上均显著优于CGFormer，平均提升超2.5点。
- **消融**：CMP带来+5.16 mIoU，SAM带来+4.3 mIoU，ICL带来+3.32 P@0.9；三种prompt缺一不可，mask prompt单独贡献+1.35 mIoU。

## 相关工作脉络
1. **Referring Image Segmentation**：从早期CNN+LSTM（MATTNet）到Transformer（LAVT、CGFormer、DMMI），本文定位于CLIP+SAM时代的新范式，区别于基于纯Transformer的交叉注意力融合方法。
2. **CLIP-based RIS（CRIS、ETRIS）**：这两个方法均采用CLIP加cross-modal decoder的两阶段方式，本文通过prompt learning保留CLIP冻结参数并桥接SAM，避免了mask分类范式的局限。
3. **Prompt Learning（CoOp、MaPLe、Co-CoOp）**：现有方法多为单向提示或仅用于图像级任务，本文扩展至像素级RIS并提出双向跨模态提示，是对prompt learning范式的扩展。
4. **Segment Anything Model（SAM）**：SAM本身为通用分割模型，本文将其作为RIS的后处理精细化模块，利用CLIP生成的prompt替代人工点提示，拓展了SAM在VL任务中的应用方式。
5. **多实例对比学习**：ICL借鉴了对比学习思想，但与标准 instance discrimination 不同，它是在同一图像的多个自然语言描述之间建立正负样本关系，更具任务针对性。

## 局限性与未来方向
- **SAM部分未完全端到端训练**：SAM的image encoder和mask decoder均被冻结，仅可学习prompt相关参数，限制了模型对分割能力的深度优化。
- **单mask预测**：SAM解码器实际可输出多个mask，本文仅使用第一个预测，可能浪费了SAM的多假设能力。
- **Gumbel-Softmax的近似性**：从mask到point坐标的转换使用Gumbel-Softmax近似，推理时仍退化为hard argmax，存在一定的精度损耗。
- **未来方向**：探索SAM的端到端微调、引入更多类型的prompt（如box prompt）、扩展到多目标RIS场景、探索其他大规模预训练模型（如Grounding DINO）的结合。

## 研究启发与可借鉴点
1. **Prompt Learning适配像素级任务**：将prompt learning从图像级（CoOp等）扩展到像素级分割的思路，可迁移至其他细粒度VL任务如图像字幕生成、物体检测等。
2. **双向跨模态提示机制**：CMP的V2L+L2V对称设计值得借鉴，可用于改进其他VLM的细粒度对齐能力，尤其适用于需要空间感知的任务。
3. **实例对比学习策略**：ICL利用同一图像的多种描述构造对比信号，可推广至多实例分割、 referring expression comprehension 等场景中增强实例判别。
4. **大模型桥接范式**：CLIP+SAM的prompt驱动桥接方式提供了一种通用的"VLM+专用模型"融合范式，可复用于其他需结合通用视觉语言知识与精细分割能力的任务。

## 关键术语表
**Referring Image Segmentation (RIS)**：根据自然语言描述在图像中分割出目标对象的视觉语言理解任务。
**Cross-Modal Prompting (CMP)**：在CLIP编码各层引入双向（图像→文本、文本→图像）可学习prompt token的跨模态交互机制。
**Instance Contrastive Learning (ICL)**：利用同一图像的多种描述构造正负样本对，通过对比损失增强模型对多实例的判别力。
**Gumbel-Softmax**：一种可微的离散采样近似方法，此处用于将连续的mask响应图转换为point prompt的位置embedding。
**Prompt-RIS**：本文提出的端到端prompt驱动框架，桥接CLIP和SAM完成RIS任务。
**oIoU / mIoU**：overall/mean Intersection over Union，RIS任务常用的分割精度评估指标。
**P@X**：IoU超过阈值X的样本比例，衡量模型在高精度要求下的表现。

## 可复现要素
- **数据集**：RefCOCO、RefCOCO+、G-Ref（均基于MS-COCO，公开可获取）
- **代码**：论文未提及代码开源状态
- **权重**：预训练的CLIP（ViT-B/16）和SAM权重（论文未明确是否公开，需自行获取）
- **关键超参**：图像分辨率（CLIP: 480×480, SAM: 1024×1024），prompt token数量n=16，batch size=64（16图×4句），训练50 epoch（前20 epoch训练CLIP+CMP+ICL，后30 epoch端到端联合训练），初始学习率1e-4，AdamW优化器，polynomial decay power=0.9，prompt dropout随机丢弃三类prompt
