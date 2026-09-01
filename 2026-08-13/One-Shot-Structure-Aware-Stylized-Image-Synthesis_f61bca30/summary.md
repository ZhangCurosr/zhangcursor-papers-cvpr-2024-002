---
title: "One-Shot-Structure-Aware-Stylized-Image-Synthesis"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Cho_One-Shot_Structure-Aware_Stylized_Image_Synthesis_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 15:19:42"
field: "图像风格化与生成"
keywords: ["one-shot stylization", "diffusion model", "structure preservation", "style transfer", "latent code disentanglement", "CLIP directional loss", "SPN"]
innovations: ["通过结构潜码和时间步控制显式解耦图像结构与语义信息", "设计结构保留网络SPN在扩散逆向过程中注入空间结构信息", "分级语义条件策略实现内容/风格的精细混合控制"]
benchmarks: ["FFHQ", "AAHQ", "MetFaces", "ArtFID", "ID Similarity", "Structure Distance"]
---

# 论文速读：One-Shot-Structure-Aware-Stylized-Image-Synthesis

## 一句话总结
OSASIS是一种基于扩散模型的一次性风格化图像合成方法，通过解耦图像的结构与语义信息，利用语义潜码的分级条件和结构潜码的时间步控制，实现了对输入图像结构的高保真保留，尤其在处理训练集中罕见结构和域外参考图像时显著优于现有方法。

## 研究问题与动机
- GAN-based风格化方法在推理时难以准确分离参考图像的**结构与风格**，导致结构伪影从参考图像渗入输出图像（如图1b所示）
- GAN方法对**罕见结构元素**（如手、麦克风等低频训练样本）的结构保真能力差，风格化后结构完整性严重偏离
- 现有扩散模型风格化方法（DiffuseIT、InST）虽然引入了扩散框架，但**未优先考虑输入图像结构的保留**，且存在域间差距（domain gap）和过度风格集中问题
- 结构保真能力并非来自扩散模型本身，而是需要特定的方法论设计

## 核心贡献（创新点）
- **提出OSASIS框架**，通过结构潜码 $\mathbf{x}_{\mathbf{t}_0}$ 和语义潜码 $\mathbf{z}_{\mathrm{sem}}$ 显式解耦图像的结构与语义信息，与GAN方法依赖隐式解耦形成本质区别
- **利用DiffAE语义编码器获取语义潜码**，并通过调整编码时间步 $\mathbf{t}_0$ 控制结构保留强度，实现内容/风格的分级条件控制（低层特征用风格语义、高层特征用内容语义），区别于InST的文本反演条件方式
- **设计结构保留网络（SPN）**，使用1x1卷积在逆向扩散过程中补充空间结构信息，缓解正向编码过程引入的噪声导致的结构损失，这是本文独有的结构保真机制
- **基于CLIP方向损失微调预训练DDIM**，通过跨域损失、域内损失和重建损失的组合桥接风格域差距，无需配对数据即可实现单张参考图像风格迁移

## 方法详解
**训练流程：**
- 从风格化域B取单张参考图像 $I_B^{\mathrm{style}}$，通过正向DDPM编码到时间步 $\mathbf{t}_0$，再经逆向DDPM生成域A的语义对齐风格图像 $I_A^{\mathrm{style}}$
- 冻结预训练DDIM $\epsilon_\theta^A$ 和DiffAE语义编码器 $\mathrm{Enc}_\phi$，复制得到 $\epsilon_\theta^B$ 并通过CLIP方向损失微调
- 输入图像 $I_A^{\mathrm{in}}$ 通过正向DiffAE编码为结构潜码 $\mathbf{x}_{\mathbf{t}_0}^{\mathrm{in}}$，其中时间步 $\mathbf{t}_0$ 可调以控制结构保留程度

**结构保留网络（SPN）：**
- 使用1x1卷积 $SPN(I_A^{\mathrm{in}})$ 提取空间结构信息
- 在逆向扩散的每一步，将SPN输出以权重 $\lambda_{SPN}$ 加到当前潜码上：$\mathbf{x}_t' = \mathbf{x}_t + \lambda_{SPN} * \mathbf{x}_t^{SPN}$
- 然后送入微调后的DDIM $\epsilon_\theta^B$ 进行去噪，最终生成风格化图像 $I_B^{\mathrm{in}}$

**损失函数（灵感来自MTG）：**
- **跨域损失（cross-domain loss）**：确保从域A到域B的变化方向一致，即 $I_A^{\mathrm{in}} \to I_B^{\mathrm{in}}$ 的方向与 $I_A^{\mathrm{style}} \to I_B^{\mathrm{style}}$ 对齐
- **域内损失（in-domain loss）**：测量域A内部和域B内部变化的相似度，防止意外变化
- **重建损失（reconstruction loss）**：将 $I_A^{\mathrm{style}}$ 编码后通过SPN生成 $\hat{I}_B^{\mathrm{style}}$，与真实 $I_B^{\mathrm{style}}$ 比较，包含 $L_1$ 损失、感知相似度损失和 $L_1$ CLIP嵌入损失

**推理（采样）策略：**
- 条件混合：将风格语义潜码 $\mathbf{z}_{\mathrm{sem}}^{\mathrm{style}}$ 条件化到UNet的低层特征图（传递风格），将输入语义潜码 $\mathbf{z}_{\mathrm{sem}}^{\mathrm{in}}$ 条件化到高层特征图（传递内容）
- 结构潜码 $\mathbf{x}_{\mathbf{t}_0}^{\mathrm{in}}$ 用于保持输入图像的整体结构完整性
- **文本驱动编辑**：直接优化输入图像的语义潜码 $\mathbf{z}_{\mathrm{sem}}^{\mathrm{in}}$（通过CLIP方向损失），再与微调后的DDIM结合进行风格化

## 实验与结果
- **数据集**：训练使用FFHQ；风格参考图像来自AAHQ、MetFaces及前作使用的风格图像（各5张）
- **低密度图像筛选**：从FFHQ随机选20,000张，通过Stochastic Reconstruction的LPIPS分数排序，选取Top 100高LPIPS（低密度）和最低LPIPS（高密度）两组进行验证
- **评估指标**：ArtFID（风格化质量）、ID Similarity（ArcFace，内容保留）、Structure Distance（结构保留）
- **主要结果（低密度图像，Table 1）**：
  - OSASIS在**ID Similarity**上全面领先：AAHQ 0.6825、MetFaces 0.7323、Prev 0.7029，显著优于最强基线DiffuseIT（0.6992/0.7158/0.6994）
  - OSASIS在**Structure Distance**上最佳：AAHQ 0.0361、MetFaces 0.0295、Prev 0.0391
  - ArtFID方面，OSASIS（34.89/43.20/33.20）优于多数基线，仅次于Prev集上的InST（35.86）
- **OOD参考图像实验**：GAN基线（MTG+HFGI、JoJoGAN+HFGI）出现严重结构伪影，OSASIS成功仅迁移风格而不引入参考图像结构
- **泛化实验**：在AFHQ-dog、LSUN-church、DeepFashion上均取得良好风格化效果
- **消融（Table 2）**：SPN使ArtFID从36.41改善至34.89，ID Similarity从0.6595提升至0.6825，Structure Distance从0.0371降至0.0361；$\lambda_{SPN}=0.1$为最优平衡点

## 相关工作脉络
- **MTG [39]**：GAN-based一次适配方法，使用CLIP方向损失微调生成器；本文沿用其损失思想但迁移至扩散模型框架，且MTG依赖过时反演方法，结构保真较弱
- **JoJoGAN [2]**：GAN-based风格映射方法，通过随机风格混合生成训练数据；本文方法无需配对或合成训练数据，直接在预训练扩散模型上微调
- **DiffuseIT [15]**：无训练扩散风格化方法，利用CLIP和ViT引导生成；本文方法通过微调弥合域间差距，风格化效果更优，但DiffuseIT在ID保留上仍有竞争力
- **InST [37]**：基于文本反演的扩散风格化方法，将风格图像概念化为文本条件；本文直接利用图像语义潜码，避免了过度风格集中和颜色转移困难的问题
- **DiffAE [23]**：提出语义编码器 $\mathrm{Enc}_\phi$ 获取线性、可解码的语义潜码；本文借用此编码器作为结构-语义解耦的基础设施
- **DiffusionCLIP [13] / Asyrp [17]**：使用CLIP方向损失实现文本驱动编辑；本文将相同损失思想应用于图像引导的风格化任务，并扩展至结构保留场景

## 局限性与未来方向
- **训练时间较长**：相比对比方法，OSASIS需要更长的微调训练时间，是结构保真能力增强的权衡
- **需为每个风格图像单独训练**：目前每个风格化任务需针对单张参考图像进行独立微调，难以快速部署多风格场景
- 未来工作将聚焦于优化训练效率、减少或对单个风格图像微调的依赖，以提升在实际场景中的实用性

## 研究启发与可借鉴点
- **结构-语义解耦范式**：通过结构潜码（时间步控制）和语义潜码（DiffAE编码器）的显式分离，为其他结构敏感生成任务（如编辑、超分）提供了可复用的设计思路
- **SPN设计简洁高效**：仅用1x1卷积在扩散逆向过程中注入空间结构信息，计算开销极小但效果显著，可迁移至其他需结构保真的扩散任务
- **低密度图像评估策略**：利用Stochastic Reconstruction的LPIPS残差筛选训练分布外样本，为模型鲁棒性评估提供了一种量化的新视角
- **分级语义条件策略**：将不同语义潜码分别条件化到UNet的高低层特征图，实现了内容/风格的精细控制，该策略可推广至多模态条件生成任务
- **无配对数据的CLIP方向微调**：仅需单张风格参考图像即可桥接域差距，结合域内损失防止过拟合，为少样本风格迁移提供了实用范式

## 关键术语表
- **One-shot Stylization（一次性风格化）**：仅需单张参考图像即可完成风格迁移的任务设定，无需大量风格配对数据
- **Structural Latent Code（结构潜码）** $\mathbf{x}_{\mathbf{t}_0}$：将图像编码到扩散过程中的特定时间步获得的潜表示，承载图像的整体结构和空间信息
- **Semantic Latent Code（语义潜码）** $\mathbf{z}_{\mathrm{sem}}$：通过DiffAE语义编码器获得的潜变量，具有线性、可解码和语义丰富特性，承载图像的内容和风格语义
- **Structure-Preserving Network (SPN)**：基于1x1卷积的轻量子网络，在扩散逆向过程中补充空间结构信息，防止边缘结构失真
- **CLIP Directional Loss**：基于CLIP嵌入空间的损失函数，通过优化生成图像与目标方向的一致性来桥接域差距或实现属性编辑
- **Out-of-Domain (OOD) Reference Image**：与输入图像风格差异较大的参考图像，传统GAN方法在此场景下易产生结构伪影
- **Low-density Image（低密度图像）**：在训练分布中罕见出现的图像，包含多样物体和遮挡，用于严格评估模型结构保真能力

## 可复现要素
- **数据集**：FFHQ（公开）、AAHQ（公开）、MetFaces（公开）、AFHQ-dog、LSUN-church、DeepFashion（均公开）
- **代码**：开源，见 https://github.com/hansam95/OSASIS
- **预训练权重**：使用公开的FFHQ预训练DDIM和DiffAE权重；基线模型使用论文提供的公开预训练权重
- **关键超参**：$\mathbf{t}_0$（编码时间步）、$\lambda_{SPN}=0.1$（SPN权重系数）、$f_{ch}$（条件切换特征层）；具体数值论文正文未详述，需在补充材料或代码中查找
