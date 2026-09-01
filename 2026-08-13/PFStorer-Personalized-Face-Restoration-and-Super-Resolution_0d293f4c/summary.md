---
title: "PFStorer-Personalized-Face-Restoration-and-Super-Resolution"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Varanka_PFStorer_Personalized_Face_Restoration_and_Super-Resolution_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 20:14:44"
field: "人脸复原与个性化生成"
keywords: ["face restoration", "personalized generation", "diffusion model", "super-resolution", "adapter", "identity fidelity"]
innovations: ["基于适配器的个性化人脸复原架构，冻结基础模型仅训练可学习门控适配器，避免灾难性遗忘", "生成式正则化机制，强制模型仅凭参考图像学习鲁棒身份表征，防止依赖LQ输入走捷径", "无对齐随机裁剪训练pipeline，结合二阶退化模型支持高分辨率超分辨率"]
benchmarks: ["Celeb-Ref", "ArcFace ID similarity", "LMSE"]
---

# 论文速读：PFStorer-Personalized-Face-Restoration-and-Super-Resolution

## 一句话总结
本文提出PFStorer，一种基于扩散模型的**个性化人脸复原与超分辨率**方法，通过使用少量高质量参考图像对基础复原模型进行微调，使模型能够在严重退化条件下保留并恢复人物身份特征，同时生成细腻的真实感细节。

## 研究问题与动机
- **身份保真性缺失**：现有生成式人脸复原方法（如GFP-GAN、CodeFormer、DR2等）能生成高视觉质量图像，但缺乏对具体人物身份的上下文理解，导致输出可能与原人物"不是同一个人"。
- **病态问题本质**：人脸复原是典型的病态逆问题，相同低质量输入可能对应多个合法的高质量输出；人类对面部细微差异极为敏感，小变化即可导致身份感知偏移。
- **参考图像利用不足**：已有参考图像方法（如DMDNet、ASFFNet）需要精确对齐或语义相似，在姿态/光照差异大时失效；而个性化生成方法（如DreamBooth、ViCo）通常改变整个模型先验，不适合直接用于复原任务。
- **训练策略局限**：传统方法依赖人脸 landmark 对齐和固定尺寸缩放，损失高分辨率细节，且对姿态变化鲁棒性差。

## 核心贡献（创新点）
1. **基于适配器的个性化复原架构**：冻结基础扩散复原模型的参数，仅训练独立的可学习适配器模块（Personalization Block），通过跨注意力机制融合参考图像特征，避免灾难性遗忘，与DreamBooth/ViCo等全模型微调方法本质不同。
2. **可学习门控向量实现逐层平衡控制**：引入可学习向量 $\gamma^i$ 动态调节每个网络层中基础模型输出与个性化特征的融合比例，精细控制个性化注入程度，解决适配器直接插入导致的中间特征突变问题。
3. **生成式正则化机制**：设计生成式正则损失 $\mathcal{L}_{Gen}$，在训练时随机以空输入替代LQ图像，强制模型仅凭参考图像学习鲁棒的身份神经表示，防止模型走捷径依赖LQ图像中的身份线索。
4. **无对齐训练流程改进**：采用随机裁剪替代人脸对齐，结合二阶噪声模型、ISP模型和运动模糊合成退化；保留HQ输入直传机制（概率0.03），提升模型对姿态变化的鲁棒性并天然支持图块拼接超分辨率。

## 方法详解
**整体框架**：以预训练的StableSR为基础模型，在其UNet各层插入Personalization Block。推理时仅需输入LQ图像（无需参考图像，身份已编码为可学习参数）；训练时输入LQ图像和随机采样的参考图像集合。

**Personalization Block设计**：
- 参考图像 $I_{Ref}$ 通过冻结的StableDiffusion提取特征 $F_{Ref}^i$
- 在每层对文本嵌入token、参考图像特征 $F_{Ref}^i$ 和中间特征 $F^i$ 执行文本-图像交叉注意力
- 输出公式：$\hat{F}^i = F^i + \gamma^i \odot \mathrm{Personalization\text{-}Block}(F^i, F_{ref}^i)$，其中 $\gamma^i$ 初始化为零向量

**损失函数**：
- 扩散损失：$\mathcal{L}_{Diff} = \mathbb{E}_{z,t,I_{LQ},I_{Ref},\epsilon}\| \epsilon - \epsilon_\theta(z_t, c, I_{LQ}, I_{Ref})\|_2$
- 生成式正则损失：$\mathcal{L}_{Gen} = \mathbb{E}_{z,t,I_{LQ},I_{Ref},\epsilon}\| \epsilon - \epsilon_\theta(z_t, c, \varnothing, I_{Ref})\|_2$（用空输入替代LQ条件）
- 总损失：$\mathcal{L} = \mathcal{L}_{Diff} + \lambda_{Gen}\mathcal{L}_{Gen} + \lambda_{Pers}\mathcal{L}_{Pers}$
- 默认 $\lambda_{Gen}=0.1$，$\mathcal{L}_{Pers}$ 正则化文本embedding的跨注意力图（防止过拟合，参考ViCo）

**Base Model改进训练**：
- 初始化：StableSR（非从头训练，利用通用图像复原先验）
- 训练数据：FFHQ人脸数据集
- 退化模拟：二阶噪声模型 + ISP模型 + 运动模糊 + 中值模糊
- 数据增强：随机裁剪（非对齐、非等比缩放），HQ直传概率 $p_{HQ}=0.03$
- 超分辨率：通过tiling方法支持 $>512\times512$ 图像的20倍超分

## 实验与结果
**数据集**：
- Celeb-Ref子集：20个身份，每个≥10张图，5张用于个性化微调，剩余242张测试（分light/heavy退化两组）
- 真实世界数据：从LAION-5B-KNN反向搜索收集20个身份的真实网络图片（远距离/失焦/弱光场景）

**评估指标**：PSNR、SSIM、LPIPS、MUSIQ(KonIQ)、LMSE（landmark MSE）、ID（ArcFace余弦相似度）

**主要定量结果（重度退化）**：
| 方法 | Ref | ID↑ | LMSE↓ | MUSIQ↑ | PSNR↑ |
|------|-----|-----|-------|--------|-------|
| CodeFormer | — | 38.33 | 33.34 | 60.92 | 22.26 |
| DMDNet | √ | 29.51 | 89.26 | 47.17 | 22.64 |
| DR2+SPAR | — | 30.01 | 40.82 | 47.36 | 22.17 |
| **PFStorer** | **√** | **57.18** | **18.37** | **64.04** | **22.62** |
| GT | — | 100 | 0 | 62.37 | ∞ |

- PFStorer在**ID指标上达57.18%**，比次优CodeFormer高出**近20个百分点**；LMSE误差降低近**一半**（18.37 vs 33.34）
- MUSIQ perceptual quality 64.04，超越所有基线

**消融实验**：
- Base Model vs Base+ViCo vs Base+DreamBooth vs PFStorer：DreamBooth微调破坏先验导致PSNR下降至21.13（低于StableSR的21.68）；ViCo虽保留PSNR但ID仅53.92%；PFStorer在ID(57.18%)和LMSE(18.37)上均最优
- 生成正则化权重影响：$\lambda_{Gen}$ 较大时促进 hallucination（适合重度退化），较小时侧重忠实复原（适合轻度退化）

**User Study**：40名用户、8000条回答，PFStorer在**身份保真度**上领先CodeFormer 36.6个百分点，整体获票率**61%**（次优25%）

## 相关工作脉络
- **GFP-GAN/CodeFormer/VQFR**：基于GAN/codebook的生成式先验方法，擅长生成 realistic texture 但缺乏 identity context；PFStorer 与之正交——不改变基础生成能力，而是通过适配器注入个性化先验。
- **DreamBooth/Custom-Diffusion/Perfusion/ViCo**：个性化生成方法，通过微调/LoRA/适配器学习特定主体；PFStorer 借鉴ViCo的adapter思想，但目标是从"个性化生成"转向"个性化复原"，且引入γ门控和生成正则化解决复原任务的特殊挑战。
- **DMDNet/ASFFNet/GFRNet**：参考图像引导的复原方法，依赖landmark对齐或特征字典匹配；PFStorer 无需对齐，学习的是抽象的identity神经表示，对姿态/光照变化更鲁棒。
- **StableSR**：通用图像超分扩散模型；PFStorer 以其为Base Model起点，利用其通用先验减少训练时间并提升鲁棒性。
- **DR2/DifFace/DiffBFR**：扩散复原方法；PFStorer 与之对比显示个性化方法在ID保真度上的显著优势。

## 局限性与未来方向
- **参考图像与输入外观不一致时输出不理想**：模型忠实于参考图像，若参考图中人物外观发生变化（如发型/年龄），复原结果可能产生 unwanted artifacts（Fig.10）。
- **扩散模型固有限制**：采样速度慢，存在随机性导致的偶尔幻觉或伪影；可结合后续工作（如[6] Emu）引导生成方向。
- **未系统评估极端语义缺失场景**：如完全遮挡、极低分辨率（<32×32）下个性化信息的可靠性有待验证。
- **未来方向**：探索无需参考图像的隐式身份建模；结合视频时序一致性做动态人脸复原；加速推理（如蒸馏/低秩分解）。

## 研究启发与可借鉴点
1. **Adapter+门控向量的"先验保护"范式**：冻结主干+可学习适配器+逐层γ门控的设计，适用于任何需要在保留预训练先验的同时注入领域/个体知识的任务（如医学图像复原、旧照片修复），是本团队可复用的核心技巧。
2. **生成式正则化的"脱离条件"训练策略**：随机以null输入替代条件信号，强制模型学习内化表征而非依赖输入线索——可迁移至其他条件生成任务（如图像补全、视频插值）中防止shortcut学习。
3. **无对齐随机裁剪训练 pipeline**：放弃landmark对齐改用随机裁剪，既提高姿态鲁棒性又保留高分辨率patch；结合HQ直传（小概率）防止细节退化，这一训练策略值得在各类复原任务中推广。
4. **个性化复原的评估范式**：本文同时报告ID(ArcFace)和LMSE，弥补了传统PSNR/SSIM无法捕捉身份保真性的缺陷；建议后续研究引入identity-aware度量作为标准评测维度。
5. **参考图像集合的灵活性**：3-5张自拍即可微调，且参考图可与输入图姿态/光照差异大；这种"few-shot reference-based personalization"模式在老照片修复、历史人物复原等场景中具有直接应用价值。

## 关键术语表
**Personalized Face Restoration**：利用特定人物的参考图像，使复原模型能生成符合该人物身份特征的高清人脸图像的技术方向。

**Adapter（适配器）**：冻结预训练模型参数后插入的可学习模块，通过残差连接注入新能力而不破坏原始先验，常用于高效微调。

**Generative Regularization（生成式正则化）**：训练时随机移除条件输入（LQ图像），迫使模型仅依靠参考图像生成内容，从而学习更鲁棒的内部表征。

**Learnable Gate Vector $\gamma^i$**：每层一个的可学习向量，控制该层基础特征与个性化特征的融合比例，初始化全零以避免训练初期输出剧变。

**Alignment-Free Training**：不依赖人脸 landmark 检测和几何对齐，改用随机裁剪处理输入，提升模型对姿态变化和检测失败的鲁棒性。

**Ill-posed Restoration Problem（病态复原问题）**：从低质量观测恢复高质量图像的逆问题存在多解性，需额外先验约束才能得到唯一合理输出。

**Tiling-based Super-Resolution**：将大图分割为重叠图块分别进行超分辨率处理再合并，支持任意尺寸图像的放大（本文实现20×超分）。

**ID Metric（ArcFace cosine similarity）**：使用预训练人脸识别模型ArcFace计算生成图像与GT图像的余弦相似度，量化身份保真程度。

## 可复现要素
- **数据集**：Celeb-Ref（论文使用子集，原数据集公开）；FFHQ（基础模型训练）；LAION-5B-KNN（真实世界参考图像检索）
- **代码/权重**：论文未明确声明开源，需关注作者主页或后续更新
- **关键超参**：个性化微调500 iterations（单张A100约10分钟）；$\lambda_{Gen}=0.1$；$p_{HQ}=0.03$；参考图像数量5张；输入分辨率基础模型基于512×512
- **基础模型**：StableSR（预训练权重公开）
