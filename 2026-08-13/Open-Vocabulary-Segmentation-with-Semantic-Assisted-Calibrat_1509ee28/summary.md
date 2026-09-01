---
title: "Open-Vocabulary-Segmentation-with-Semantic-Assisted-Calibrat"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_Open-Vocabulary_Segmentation_with_Semantic-Assisted_Calibration_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 15:19:52"
field: "开放词汇语义分割"
keywords: ["Open-Vocabulary Segmentation", "CLIP", "Semantic Integration", "Contextual Shift", "Vision-Language Alignment"]
innovations: ["通过CLIP全局语义先验校准proposal embedding避免in-vocabulary坍塌", "上下文迁移策略缓解裁剪掩码图像的域偏移问题", "提出考虑类别语义关系的SG-IoU评估指标"]
benchmarks: ["ADE20K-150", "ADE20K-847", "Pascal Context-59", "Pascal Context-459", "Pascal VOC"]
---

# 论文速读：Open-Vocabulary-Segmentation-with-Semantic-Assisted-Calibration

## 一句话总结
提出SCAN网络，通过CLIP的全局语义先验校准proposal embedding（避免坍塌到已知类别）和domain-biased CLIP分类空间（缓解裁剪掩码图像的域偏移），在多个开放词汇分割基准上达到SOTA。

## 研究问题与动机
1. **Proposal embedding坍塌到in-vocabulary**：分割模型的可学习query embedding倾向于拟合训练语义空间，对未见过的新类别不敏感，导致分类坍塌。
2. **CLIP输入存在domain bias**：两阶段方法通过裁剪掩码生成子图输入CLIP，这类人工构造的图像偏离自然图像分布，丢失全局上下文并引入错误背景先验。
3. **现有评估指标忽视语义冗余**：ADE20K等数据集存在大量语义重复类别（如"chair"和"armchair"同时存在），传统mIoU对这类细粒度层次区分过于严苛，不适合开放词汇场景。
4. **两阶段方法整体性能仍低于监督方法**：尽管利用CLIP作为额外分类器取得进展，但在相关场景下性能仍远低于监督分割模型。

## 核心贡献（创新点）
1. **SCAN框架**：系统性校准in-vocabulary和out-of-vocabulary语义空间，显著提升视觉-语言对齐能力；本质区别于已有工作在于同时校准proposal embedding和CLIP输入分布，而非仅依赖集成分类结果。
2. **语义集成模块（SIM）**：将CLIP的全局[CLS]嵌入和多层空间token通过低通滤波增强后注入proposal embedding，扩展语义空间；与OVSeg等方法的不同在于利用频谱增强抑制高频噪声，并桥接视觉与语言统一空间。
3. **上下文迁移策略（CS）**：在CLIP前向传播中随机替换掩码背景区域为原始图像[CLS] embedding，缓解域偏移和错误背景先验；与MaskCLIP等方法的区别在于不需要finetune CLIP架构，而是通过特征层面的上下文补充实现域适应。
4. **新评估指标SG-IoU**：考虑类别间语义关系（同义词、上下位词），对open-vocabulary场景更友好；区别于传统mIoU，允许预测为父类或同义词时视为正确。

## 方法详解
**整体框架**：基于两阶段范式，先用分割模型生成class-agnostic mask proposals（$M_N \in \mathbb{R}^{N \times H \times W}$）和proposal embeddings（$F_N \in \mathbb{R}^{N \times C}$），再通过SIM和CS两个模块分别校准两类特征空间。

**语义集成模块（SIM）**：
- 从冻结CLIP提取渐进特征$\{F_{HW}^i, F_{CLS}^i\}$，利用空间token和全局[CLS] token双重信息。
- 对选定的CLIP层空间特征进行低通滤波增强（基于频域理论，低频分量对应高层语义）：
  $$g^i = \text{Gaussian}(h, w, \sigma), \quad F_s^i = \text{FFT}(F_{HW}^i) * g, \quad F_s^i = \text{IFFT}(\text{ReLU}(\text{Conv}(F_s^i))) + F_{HW}^i$$
- 通过多头交叉注意力将$F_s$注入proposal embedding：$F_N' = \text{MHA}(F_N, F_s, F_s)$
- 利用对齐的CLIP最终[CLS]嵌入桥接视觉-语言空间：$F_N^{final} = \text{Trans.Enc.}(F_N' + \gamma * F_{CLS}^{final})$

**上下文迁移策略（CS）**：
- 在CLIP前向传播时，对选定层的背景区域随机替换$\alpha\%$为原始图像的$F_{CLS}^{i-1}$：
  $$F_m^i = \mathcal{V}^i(F_m^{i-1} | (M_k, F_{CLS}^{i-1}, \alpha)), \quad i \in \text{idx}$$
- 双重作用：① 提供全局上下文缓解域偏移；② 破坏规则背景形状，削弱错误分布先验。
- CLIP仍遵循OVSeg在masked images dataset上进行finetune。

**分类融合**：整合proposal embedding和CLIP的mask-level分类结果，输出open-vocabulary类别标签。

## 实验与结果
- **训练数据**：COCO-Stuff（171类）训练分割模型；CLIP在OVSeg的masked images dataset（来自COCO Captions）上微调。
- **评测基准**：ADE20K-150、ADE20K-847、Pascal VOC、Pascal Context-59（PC-59）、Pascal Context-459（PC-459）。
- **最强结果（CLIP ViT-L/14）**：
  - ADE-150: **33.5** mIoU（超SAM 32.1约1.4个百分点）
  - ADE-847: **14.0** mIoU（超SAN 12.4约1.6个百分点）
  - PC-59: **59.3** mIoU
  - PC-459: **16.7** mIoU
  - VOC: **97.2** mIoU
- **消融实验**：SIM单独贡献+1.4（ADE-150），CS单独贡献+1.3，两者结合达最高；最佳层选择为12/18/24层，kernel为9/7/3；CS最佳替换层为1/3/5/7/9，比例约30%。
- **SG-IoU结果**：SCAN在ADE-150达34.2，较mIoU提升0.7个百分点，充分验证语义关系纳入评估的合理性。

## 相关工作脉络
1. **两阶段OVS基线**：OpenSeg、SimSeg、OVSeg、ODISE等均采用"分割模型生成proposals + CLIP辅助分类"范式，本文与其定位差异在于通过校准而非简单集成来克服两类空间的局限。
2. **SAM**：通过side-adapter解耦分割与分类，但仍受限于learnable参数带来的bounded语义空间；SCAN利用CLIP的全局先验进行特征校准。
3. **GKC**：通过文本引导的知识蒸馏提升效率，本文聚焦精度与语义校准。
4. **MaskCLIP**：探索open-vocabulary panoptic segmentation，本文专注semantic segmentation并引入新评估指标。
5. **CLIP视觉-语言预训练**：作为核心组件，本文创新性地将其[CLS] token用于特征空间校准和域偏移缓解。

## 局限性与未来方向
1. **CLIP finetune依赖**：CS策略仍需对CLIP进行masked image微调（沿袭OVSeg），限制了完全zero-shot的设置。
2. **层选择敏感**：SIM和CS的最优层选择需要实验调参（如表5、表7所示），缺乏自动选择机制。
3. **替换比例权衡**：高比例替换可能损害局部区域判断，比例选择需折中。
4. **SG-IoU的通用性**：语义关联矩阵需人工构建，不同数据集的类别层次关系不同，泛化到跨数据集场景有待探索。

## 研究启发与可借鉴点
1. **频域低通滤波用于特征增强**：SIM中利用频谱理论抑制高频噪声、提取高层语义的设计，可迁移到其他视觉-语言对齐任务中。
2. **全局[CLS] token的多次利用**：不仅用于分类，还作为空间特征的补充先验和背景替换源，体现了CLIP表征的多层次利用价值。
3. **域偏移的特征层面缓解**：CS策略通过替换token而非重训网络来缓解域偏移，为vision-language模型在人工构造输入上的适配提供了新思路。
4. **评估指标的语义感知设计**：SG-IoU对类别冗余问题的处理，可推广到其他open-world视觉任务（如detection、panoptic segmentation）的评估体系设计。

## 关键术语表
**Open-Vocabulary Segmentation (OVS)**：给定任意文本查询而非固定类别集合的语义分割任务，支持对训练集中未见类别的识别。
**Semantic Integration Module (SIM)**：将CLIP的全局语义先验（[CLS]嵌入和多尺度空间token）注入proposal embedding以扩展语义空间的结构。
**Contextual Shift (CS)**：在CLIP前向传播中随机用原始图像[CLS]嵌入替换掩码背景区域，以缓解域偏移和错误背景先验的策略。
**Low-Frequency Enhancement**：基于频域理论的特征过滤方法，保留低频分量以抑制纹理噪声、增强高层语义感知。
**SG-IoU (Semantic-Guided IoU)**：考虑类别间语义关系（同义词、上下位词）的开放词汇分割评估指标。
**In-vocabulary Collapse**：proposal embedding因学习参数限制而过度拟合训练类别，对未见过类别不敏感的现象。
**Domain Bias**：裁剪掩码图像偏离自然图像分布，导致预训练CLIP分类性能下降的问题。

## 可复现要素
- **数据集**：COCO-Stuff（训练，公开）、COCO Captions（CLIP微调用，公开）、ADE20K、Pascal VOC、Pascal Context（评测，公开）。
- **代码**：论文声明Code available here（已开源）。
- **权重**：使用OpenCLIP的ViT-B/16和ViT-L/14，权重公开。
- **关键超参**：batch size=32，训练迭代120k，基础学习率0.00006，输入最短边resize到640，loss权重5（分割）和2（分类），CS替换比例α=30%，SIM选择CLIP层12/18/24，kernel 9/7/3。
