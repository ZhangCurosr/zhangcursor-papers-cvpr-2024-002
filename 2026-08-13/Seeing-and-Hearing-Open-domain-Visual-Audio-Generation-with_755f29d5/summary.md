---
title: "Seeing-and-Hearing-Open-domain-Visual-Audio-Generation-with"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Xing_Seeing_and_Hearing_Open-domain_Visual-Audio_Generation_with_Diffusion_Latent_Aligners_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 20:15:16"
field: "多模态生成"
keywords: ["跨模态生成", "视频音频联合生成", "扩散模型", "多模态对齐", "训练-free方法", "ImageBind"]
innovations: ["提出Diffusion Latent Aligner在扩散去噪过程中注入跨模态梯度指导", "首次实现文本引导的开放域联合视频-音频生成", "利用预训练ImageBind作为训练-free的跨模态对齐桥梁"]
benchmarks: ["VGGSound", "Landscape"]
---

# 论文速读：Seeing-and-Hearing-Open-domain-Visual-Audio-Generation-with-Diffusion-Latent-Aligners

## 一句话总结
本文提出一种**训练-free的跨模态对齐框架**，利用预训练的ImageBind模型在扩散latent空间中建立视觉-音频共享语义桥梁，通过梯度引导同时实现视频到音频、音频到视频、图像到音频及联合视觉-音频生成，无需大规模配对数据集与额外网络训练。

## 研究问题与动机
- **单模态生成割裂**：现有扩散模型分别独立生成视频或音频，无法产出语义对齐的视听内容，难以支撑影视制作等需要多模态协同的创作场景。
- **已有跨模态方法局限**：现有V2A/A2V方法（如SpecVQGAN、TempoTokens）受限于特定领域或生成质量较低；联合生成工作（如MM-Diffusion）仅支持无条件生成且领域受限。
- **独立模型能力强但互不相通**：预训练的文本到视频（如AnimateDiff）和文本到音频（如AudioLDM）模型各自表现优异，但缺乏跨模态协同机制。
- **ImageBind的多模态绑定能力**：ImageBind已在共享嵌入空间中对齐多种模态（图像、视频、音频、文本等），可作为跨模态bridging的基础。

## 核心贡献（创新点）
- **提出Diffusion Latent Aligner新范式**：在扩散去噪过程中动态注入跨模态梯度指导，将孤立单模态生成模型有机连接，与从头训练全新生成模型（如MM-Diffusion）形成本质区别。
- **训练-free的跨模态对齐机制**：利用预训练ImageBind计算latent与条件模态在共享空间中的距离，无需任何大规模视听配对数据微调，而现有方法大多依赖成对数据训练。
- **首次实现文本引导的联合视频-音频生成**：提出Joint-VA任务的首个开放域方案，通过双向guidance同时优化视频与音频latent及提示embedding。
- **Dual/Triangle Loss设计**：针对音频语义信息不足的问题，引入文本embedding作为补充监督信号，形成三角对齐损失，提升引导稳定性。
- **Guided Prompt Tuning**：通过优化文本提示embedding来提升音频到视频生成的对齐效果，解决帧间时序一致性问题。

## 方法详解

### 3.2.1 问题形式化
给定模态$M_1$（生成）和$M_2$（条件），扩散latent aligner $\mathcal{A}$在任意 timestep $t$ 接收当前 noisy latent $\mathbf{z}_t^{M_1}$ 和条件 $\mathbf{x}^{M_2}$，输出对齐后的 $\hat{\mathbf{z}}_t^{M_1}$：
$$\hat{\mathbf{z}}_t^{M_1} = \mathcal{A}(\mathbf{z}_t^{M_1}, \mathbf{x}^{M_2})$$
联合生成时，aligner同时接收两模态latent并输出双向指导：$(\hat{\mathbf{z}}_t^{M_1}, \hat{\mathbf{z}}_t^{M_2}) = \mathcal{A}(\mathbf{z}_t^{M_1}, \mathbf{z}_t^{M_2})$。

### 3.2.2 多模态Guidance（核心机制）
从去噪预测的$\hat{\epsilon}$反推clean latent $\tilde{\mathbf{z}}_0$：
$$\tilde{\mathbf{z}}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}} \mathbf{z}_t - \sqrt{\frac{1-\bar{\alpha}_t}{\bar{\alpha}_t}} \hat{\epsilon}$$
将$\tilde{\mathbf{z}}_0$与条件输入ImageBind编码器，计算embeddings距离作为loss：
$$\mathcal{L}(\tilde{\mathbf{z}}_0, \mathbf{x}^{M_2}) = 1 - \mathcal{F}(\mathbf{E}^{M_1}(\tilde{\mathbf{z}}_0), \mathbf{E}^{M_2}(\mathbf{x}^{M_2}))$$
其中$\mathcal{F}$为余弦距离（$1 - \text{cosine\_similarity}$）。通过backprop更新latent：
$$\hat{\mathbf{z}}_t = \mathbf{z}_t - \lambda_1 \nabla_{\mathbf{z}_t} \mathcal{L}(\mathcal{D}(\tilde{\mathbf{z}}_0), \mathbf{x}^{M_2})$$
该过程每个denoising step执行一次，仅需极小额外计算开销。

### 3.2.3 Dual/Triangle Loss
针对音频语义稀疏问题，引入文本embedding $\mathbf{e}_p$构建更全面的对齐信号：
- **A2V**：$\mathcal{L}_{a2v} = \mathcal{F}(\mathbf{e}_v, \mathbf{e}_a) + \mathcal{F}(\mathbf{e}_v, \mathbf{e}_p)$
- **V2A**：$\mathcal{L}_{v2a} = \mathcal{F}(\mathbf{e}_a, \mathbf{e}_v) + \mathcal{F}(\mathbf{e}_a, \mathbf{e}_p)$
- **Joint-VA**：$\mathcal{L}_{\text{joint-va}} = \mathcal{F}(\mathbf{e}_v, \mathbf{e}_p) + \mathcal{F}(\mathbf{e}_v, \mathbf{e}_a) + \mathcal{F}(\mathbf{e}_a, \mathbf{e}_p)$

### 3.2.4 Guided Prompt Tuning
对于音频引导的视频生成，同步优化prompt embedding $\mathbf{y}$：
$$\hat{\mathbf{y}} = \mathbf{y} - \lambda_2 \nabla_{\mathbf{y}} \mathcal{L}$$
在noise prediction前detach prompt embedding，保留从text embedding到multimodal loss的计算图，梯度更新后的embedding在整个denoising过程中共享，确保时序一致性。

### Algorithm 1 联合生成流程
在前$K$步warmup后，每个timestep执行$N$次optimization step：
1. 双端denoise得到$\mathbf{z}_t^v, \mathbf{z}_t^a$
2. 反推$\tilde{\mathbf{z}}_0^v, \tilde{\mathbf{z}}_0^a$
3. 经ImageBind得$\mathbf{e}_v, \mathbf{e}_a, \mathbf{e}_p$
4. 计算三角形loss并分别对$\mathbf{z}_t^v, \mathbf{z}_t^a, \mathbf{y}$做梯度更新

## 实验与结果

### 数据集
- **VGGSound** [6]：3k样本用于V2A、3k用于A2V、3k用于I2A（提取关键帧）
- **Landscape** [36]：200对用于Joint-VA评估

### 实现细节
- 基座模型：AudioLDM [29]（V2A/I2A）、AnimateDiff [18]（A2V/Joint-VA）
- Denoising steps：V2A=30，A2V=25，Joint-VA=25
- 学习率：$\lambda_1=0.1$（AudioLDM），$\lambda_1=0.01$（AnimateDiff），$\lambda_2=0.1$（prompt tuning）
- 硬件：NVIDIA RTX 3090

### 主要结果（Table 1）

| 任务 | Baseline | 指标 | 本文方法 |
|------|----------|------|----------|
| **V2A** | SpecVQGAN | KL↓ | 3.290 → **2.619** |
| | | ISc↑ | 5.108 → **5.831** |
| | | FAD↓ | 7.736 → **7.316** |
| **I2A** | Im2Wav | KL↓ | 2.612 → **2.691** |
| | | FAD↓ | 7.576 → **6.869** |
| **A2V** | TempoTokens | FVD↓ | 1866.285 → **402.385** |
| | | KVD↓ | 389.096 → **34.764** |
| | | AV-align↑ | 0.423 → **0.522** |
| **Joint-VA (Open-domain)** | Ours-vanilla | AV-align↑ | 0.226 → **0.283** |
| | | VT-align↑ | 0.322 → **0.324** |
| | | AT-align↑ | 0.081 → **0.138** |
| **Joint-VA (Landscape)** | MM-Diffusion + Ours | FAD↓ | 7.752 → **6.463** |

- **最强提升**：A2V任务FVD从1866降至402（**提升约4.6倍**），KVD从389降至35（**提升约11倍**）；AV-align从0.423提升至0.522。
- **训练-free优势**：在V2A和I2A任务中，无训练方法超越了需要大规模视听配对数据训练的SpecVQGAN和Im2Wav。

## 相关工作脉络

- **SpecVQGAN [26]**：基于ResNet-50特征提取的V2A方法，使用预训练codebook+Vocoder重建波形，领域受限；本文方法无需配对数据训练，泛化性更强。
- **Im2Wav [37]**：利用CLIP+LLM生成音频token再上采样，依赖语言模型链式生成；本文直接利用ImageBind空间梯度对齐，更高效。
- **TempoTokens [48]**：纯音频到视频的生成方法，视觉质量差且对齐不足；本文在保持视觉质量的同时显著改善对齐。
- **MM-Diffusion [36]**：首个无条件联合生成框架，但仅在训练域内有效；本文拓展至开放域文本引导场景并提升对齐。
- **AudioLDM [29] / AnimateDiff [18]**：各自为强单模态生成模型，本文通过Latent Aligner将其桥接为协同系统，非独立改进各自模型。
- **ImageBind [17]**：多模态绑定表征模型，本文首次将其用于diffusion生成过程中的在线对齐指导，而非仅作为离线表征工具。

## 局限性与未来方向

- **训练-free方法的精度上限**：依赖预训练模型的表示能力，在复杂语义理解场景下可能不如端到端训练方法。
- **时序一致性依赖prompt tuning**：音频引导视频生成中，帧间一致性需通过prompt优化间接保障，缺乏显式时序建模。
- **ImageBind的计算开销**：每一步去噪均需forward ImageBind并backprop，推理速度受限。
- **视频到音频的语义保真**：视频富含语义但音频语义稀疏，单向对齐可能存在信息损失。
- **未来方向**：可探索引入时序consistency loss、结合LLM增强语义理解、扩展至更多模态（如depth、thermal）。

## 研究启发与可借鉴点

- **Latent-space cross-modal guidance思路可迁移**：ImageBind等现成多模态表征模型可作为"off-the-shelf"对齐器，适用于其他模态组合（如视觉-深度、音频-文本）。
- **训练-free优化策略降低数据依赖**：无需大量配对数据即可实现跨模态对齐，适合数据稀缺或新模态组合场景。
- **Dual/Triangle loss设计模式**：利用第三模态（文本）弥补主条件模态信息不足的策略，可推广至其他不对等条件生成任务。
- **Prompt tuning与latent optimization联合优化**：同时优化输入embedding和中间latent的梯度更新策略，可在保持生成质量的同时增强条件对齐。
- **与本团队方向结合机会**：该方法可移植到视频-语言联合生成、3D-音频同步等任务，或与其他多模态绑定模型（如CLIP、BLIP-2）结合探索新应用。

## 关键术语表

**Diffusion Latent Aligner**：在扩散模型去噪过程中动态注入跨模态梯度指导的模块，利用预训练多模态模型计算latent与条件的对齐距离并反向传播更新。

**ImageBind**：Facebook提出的多模态绑定模型，将图像、视频、音频、文本、深度、热成像等模态映射到统一嵌入空间，支持跨模态检索与对齐。

**Classifier Guidance**：传统扩散模型条件生成方法，通过训练分类器计算梯度来引导去噪过程朝向目标类别；本文方法类比此思想但应用于多模态latent空间。

**Joint-VA Generation**：联合视觉-音频生成任务，从单一文本提示同时生成语义对齐的视频和音频内容。

**VGGSound**：大规模视听数据集，包含约30万段视频及其对应音频标注，广泛用于视听相关任务评测。

**FAD (Frechet Audio Distance)**：衡量生成音频与真实音频分布距离的评估指标，越低表示音质越高。

**AV-align**：音频-视频对齐度量，基于ImageBind计算的跨模态相似度，用于评估生成视听内容的同步性。

## 可复现要素

- **数据集**：VGGSound（公开）、Landscape（公开）；本文随机采样3k/200对样本进行评估
- **代码开源**：项目网站 https://yzxing87.github.io/Seeing-and-Hearing/；代码开源状态论文未明确声明
- **模型权重**：使用预训练AudioLDM [29]、AnimateDiff [18]、ImageBind [17]（均已开源）
- **关键超参**：$\lambda_1=0.1/0.01$，$\lambda_2=0.1$，denoising steps=25~30，optimization steps=N，warmup steps=K（论文未明确给出N/K具体数值）
- **硬件**：NVIDIA RTX 3090
