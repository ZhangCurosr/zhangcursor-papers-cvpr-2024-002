---
title: "Seeing-and-Hearing-Open-domain-Visual-Audio-Generation-with"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Xing_Seeing_and_Hearing_Open-domain_Visual-Audio_Generation_with_Diffusion_Latent_Aligners_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 16:38:19"
field: "多模态生成与对齐"
keywords: ["多模态生成", "视频-音频联合生成", "扩散模型", "潜在对齐", "ImageBind", "无训练方法"]
innovations: ["提出扩散潜在对齐器，在 latent 空间引入跨模态梯度指导去噪", "首个文本引导的开放域视频-音频联合生成框架", "设计三角形损失与引导式提示调优解决语义稀疏问题"]
benchmarks: ["VGGSound", "Landscape", "MM-Diffusion", "SpecVQGAN", "AudioLDM", "AnimateDiff"]
---

# 论文速读：Seeing-and-Hearing-Open-domain-Visual-Audio-Generation-with-Diffusion-Latent-Aligners

## 一句话总结
本文提出一种**优化驱动的无训练方法**，利用预训练多模态模型 **ImageBind** 作为潜在对齐器（latent aligner），在扩散模型的隐式潜在空间中桥接独立训练的视觉与音频生成模型，实现视频-音频联合生成、视频到音频、音频到视频、图像到音频等四大任务的高质量跨模态对齐生成。

---

## 研究问题与动机

- **现有方法割裂**：当前 diffusion-based 视频生成（如 AnimateDiff、VideoCrafter）和音频生成（如 AudioLDM、Make-an-Audio）模型各自独立发展，缺乏跨模态的联合生成能力。
- **两阶段方案缺陷**：先 T2V 再生成音频的两阶段 pipeline（V2A 或 A2V）依赖特定域模型，通用性差且误差累积。
- **联合生成研究稀缺**：Joint-VA（联合视频-音频生成）工作极少，已有方法（如 MM-Diffusion）仅支持无条件生成，且局限于训练域（如 Landscape）。
- **资源瓶颈**：从头训练大模型成本高昂，亟需一种轻量、可复用已有预训练模型的对齐方案。

---

## 核心贡献（创新点）

- **扩散潜在对齐器（Diffusion Latent Aligner）**：首次在 diffusion 去噪过程中引入跨模态对齐信号，通过 ImageBind 空间指导潜在变量更新，而非在像素/波形空间后处理。
- **无训练桥接范式**：无需额外训练视觉或音频生成模型，直接复用预训练的 AnimateDiff（视频）和 AudioLDM（音频），显著降低算力需求。
- **双向联合生成机制**：针对 Joint-VA 任务设计三角形损失（text-video + text-audio + video-audio），实现语义一致的视频-音频同步生成，为首个文本引导的联合生成工作。
- **引导式提示调优（Guided Prompt Tuning）**：通过反向传播优化文本嵌入向量 $\hat{y}$，弥补纯视觉/音频条件语义信息不足的缺陷，提升 A2V 任务的对齐效果。
- **多任务通用性**：统一框架支持 V2A、I2A、A2V、Joint-VA 四类任务，且在不同数据集上均优于有训练 baseline。

---

## 方法详解

### 核心架构：Diffusion Latent Aligner

1. **基础扩散模型**：采用 Latent Diffusion Model（LDM），前向加噪过程 $q(z_t|z_{t-1}) = \mathcal{N}(z_t; \sqrt{1-\beta_t}z_{t-1}, \beta_t I)$，去噪网络 $\epsilon_\theta(z_t, t, p)$ 预测噪声。

2. **ImageBind 作为对齐器**：
   - 利用 ImageBind 学习到的多模态联合嵌入空间（binding image, video, audio, text 等）。
   - 在每一步去噪中，从当前 latent $z_t$ 重建干净 latent $\tilde{z}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}z_t - \sqrt{\frac{1-\bar{\alpha}_t}{\bar{\alpha}_t}}\hat{\epsilon}$。
   - 将 $\tilde{z}_0$ 和条件模态 $x^{M_2}$ 分别编码为 $e_{M_1}, e_{M_2}$，计算 ImageBind 空间中的距离损失：$\mathcal{L} = 1 - \mathcal{F}(E^{M_1}(\tilde{z}_0), E^{M_2}(x^{M_2}))$，其中 $\mathcal{F}$ 为 1 - cosine similarity。

3. **潜在变量优化**：
   $$\hat{z}_t = z_t - \lambda_1 \nabla_{z_t} \mathcal{L}(\mathcal{D}(\tilde{z}_0), x^{M_2})$$
   在去噪步骤中注入梯度信号，引导 $z_t$ 向与条件对齐的方向演化。

4. **联合生成损失（三角形）**：
   $$\mathcal{L}_{\text{joint-va}} = \mathcal{F}(e_v, e_p) + \mathcal{F}(e_v, e_a) + \mathcal{F}(e_a, e_p)$$
   同时优化 video、audio、text 三者间的对齐。

5. **Dual/Triangle 损失设计**：
   - **V2A**：$\mathcal{L}_{v2a} = \mathcal{F}(e_a, e_v) + \mathcal{F}(e_a, e_p)$
   - **A2V**：$\mathcal{L}_{a2v} = \mathcal{F}(e_v, e_a) + \mathcal{F}(e_v, e_p)$
   - **Joint-VA**：上述三角形损失

6. **引导式提示调优**：
   当音频条件语义稀疏时，额外优化 prompt embedding：
   $$\hat{y} = y - \lambda_2 \nabla_y \mathcal{L}$$
   保留计算图从文本嵌入到多模态损失，反向传播更新共享提示。

---

## 实验与结果

### 数据集与设置
- **VGGSound**：3k 视频-音频对用于 V2A/I2A/A2V 评测。
- **Landscape**：200 对用于 Joint-VA 评测。
- **基线模型**：SpecVQGAN（V2A）、Im2Wav（I2A）、TempoTokens（A2V）、MM-Diffusion（Joint-VA）。
- **本方法基座**：AudioLDM（音频生成）、AnimateDiff（视频生成），denoising step 设为 25-30。

### 主要结果（表 1）

| 任务 | 方法 | 关键指标 | 提升 |
|------|------|---------|------|
| **V2A** | SpecVQGAN | KL=3.290, ISc=5.108, FD=37.269, FAD=7.736 | — |
| | Ours（ours） | **KL=2.619**, ISc=5.831, **FD=32.920**, FAD=7.316 | KL↓15%, FD↓12% |
| **I2A** | Im2Wav | KL=2.612, ISc=7.055 | — |
| | Ours | **KL=2.691**, ISc=6.149, FD=20.958, FAD=6.869 | FAD↓9% |
| **A2V** | TempoToken | FVD=1866.285, KVD=389.096, AV-align=0.423 | — |
| | Ours | **FVD=402.385**, **KVD=34.764**, **AV-align=0.522** | FVD↓78%, AV-align↑23% |
| **Joint-VA** | MM-Diffusion | FVD=1141.009, KVD=135.368, FAD=7.752 | — |
| | Ours+MM | FVD=1174.856, KVD=135.422, **FAD=6.463** | FAD↓17% |
| | Open-domain Ours | AV-align=**0.283**, VT-align=0.324, AT-align=0.138 | 首个开放域 Joint-VA 结果 |

### 结论
- 无训练方法显著优于有训练 baseline（如 A2V 任务 FVD 从 1866 降至 402）。
- Joint-VA 任务首次实现文本引导的开放域联合生成，AV-align 达 0.283。
- 定性结果（Fig. 3-5）显示本方法能生成语义对齐、高保真的跨模态内容，减少背景噪声和不相关声音。

---

## 相关工作脉络

- **AudioGen / Diff-Sound / Make-An-Audio / AudioLDM**：文本到音频生成，使用自回归或潜在扩散，但仅单模态。
- **SpecVQGAN / Im2Wav**：视频/图像到音频，依赖特定域预训练或 codebook，泛化受限。
- **TempoTokens / Sound2sight**：音频到视频生成，但视频多样性或时序一致性不足。
- **MM-Diffusion**：首个无条件视频-音频联合生成框架，但仅限训练域且无法文本控制。
- **MovieFactory**：基于 LLM 扩展脚本后检索生成，依赖音频库检索，非真正生成。
- **ImageBind**：多模态嵌入空间学习，本文核心对齐工具，绑定 image/video/audio/text 等模态。
- **Classifier Guidance**：传统条件生成方法，本文类比其在 latent 空间的对齐思想。

---

## 局限性与未来方向

- **计算开销**：每一步去噪需额外前向 ImageBind 计算和梯度回传，推理速度略慢于基线。
- **时间一致性**：A2V 任务中逐帧梯度更新可能导致时序不连贯，需引入 temporal prior。
- **音频语义稀疏性**：纯背景音乐等条件信息不足时对齐效果受限，依赖 prompt tuning 补偿。
- **扩展性待验证**：目前仅验证 video-audio 对，未测试 image-video-audio 三模态或更多模态。
- **未来方向**：探索更高效的对齐策略（如蒸馏）、结合 temporal diffusion、扩展至 3D/VR 多模态生成。

---

## 研究启发与可借鉴点

- **无训练对齐范式**：利用预训练多模态模型（如 ImageBind、CLIP）作为外部监督信号，避免端到端训练，可迁移至其他跨模态任务（如 audio-3D、video-text）。
- **潜在空间梯度注入**：在 diffusion 去噪过程中直接优化 latent 变量而非输出，保持生成模型原有分布特性，方法通用。
- **三角形损失设计**：通过多对多约束（text-video-audio 两两距离）解决单向条件信息稀疏问题，适用于任意多模态对齐。
- **Prompt Tuning 扩展**：当某一模态语义不足时，优化另一模态的输入 embedding 作为补充，可结合 retriever 使用。
- **开源代码复用**：基于 AudioLDM 和 AnimateDiff 等成熟基座快速搭建，降低实验门槛。

---

## 关键术语表

- **Diffusion Latent Aligner**：在 diffusion 潜在空间中利用多模态嵌入距离指导去噪过程的对齐模块。
- **ImageBind**：Meta 提出的多模态嵌入模型，将 image/video/audio/text 等绑定到同一语义空间。
- **Joint Video-Audio (Joint-VA) Generation**：同时生成视频和音频且保持语义对齐的任务。
- **Classifier Guidance**：利用分类器梯度指导无条件扩散模型生成特定类别样本的技术。
- **VGGSound**：大规模视频-音频配对数据集，含 31 万短视频及对应音频。
- **FVD / KVD**：Frechet Video Distance / Kernel Video Distance，评估生成视频质量的度量。
- **AV-align**：Audio-Video Alignment，衡量生成音频与视频同步性的指标。
- **Latent Diffusion Model (LDM)**：在压缩潜在空间执行扩散过程的生成模型，如 Stable Diffusion。

---

## 可复现要素

- **数据集**：VGGSound（公开）、Landscape（公开），评测集为随机采样 3k/200 对。
- **代码开源**：项目网站 https://yzxing87.github.io/Seeing-and-Hearing/，含补充材料。
- **基座模型**：AudioLDM（开源）、AnimateDiff（开源）、ImageBind（开源）。
- **关键超参**：denoising steps=25-30，learning rate $\lambda_1=0.1$（AudioLDM）、$0.01$（AnimateDiff），warmup steps $K$ 未明确，优化步数 $N$ 未明确。
- **硬件**：NVIDIA GeForce RTX 3090 GPU。

---
