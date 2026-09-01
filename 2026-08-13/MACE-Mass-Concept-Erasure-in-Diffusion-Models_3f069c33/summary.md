---
title: "MACE-Mass-Concept-Erasure-in-Diffusion-Models"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Lu_MACE_Mass_Concept_Erasure_in_Diffusion_Models_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 12:26:28"
field: "生成模型安全与可控编辑"
keywords: ["概念擦除", "扩散模型", "模型编辑", "LoRA", "交叉注意力", "Generative AI Safety"]
innovations: ["闭式交叉注意力优化消除残留概念信息", "概念聚焦重要性采样保护擦除特异性", "独立LoRA模块闭式融合支持百级概念擦除"]
benchmarks: ["CIFAR-10", "I2P", "MS-COCO", "GIPHY Celebrity Detector", "Image Synthesis Style Studies Database"]
---

# 论文速读：MACE-Mass-Concept-Erasure-in-Diffusion-Models

## 一句话总结
MACE 提出了一种面向大规模概念擦除的微调框架，能够从文本到图像扩散模型中同时擦除多达100个不想要的概念，同时通过在闭式交叉注意力优化与 LoRA 微调的结合，实现了泛化性（擦除同义词）与特异性（保留无关概念）的有效平衡。

## 研究问题与动机
1. **擦除数量受限**：现有概念擦除方法通常只能同时处理少于5个概念，难以扩展到大数量规模（如100个），且在大规模擦除时性能显著下降。
2. **泛化性与特异性难以兼顾**：现有方法要么无法彻底擦除概念同义词（泛化性不足），要么在擦除目标概念时过度损害了无关概念的保留（特异性不足）。
3. **注意力机制导致残留信息**：目标短语的信息会通过注意力机制嵌入到提示词的其他词汇中，仅针对目标短语本身进行擦除无法彻底消除概念残留。
4. **早期去噪步骤微调损害特异性**：对扩散模型早期去噪步骤（t > t0）进行微调会改变样本生成轨迹，导致与目标概念共享词根的其他概念（如同名人物）被误擦除。
5. **多概念擦除的灾难性遗忘与干扰**：顺序微调易引发灾难性遗忘，并行微调则导致不同概念之间的 LoRA 模块相互干扰。

## 核心贡献（创新点）
1. **闭式交叉注意力优化（Closed-Form Cross-Attention Refinement）**：通过求解闭式解优化交叉注意力的投影矩阵，将共现词汇映射到泛化类别空间，消除目标短语的残留信息；与已有方法相比，首次系统性地解决了"信息嵌入到共现词汇中"的问题。
2. **概念聚焦重要性采样（Concept-Focal Importance Sampling, CFIS）**：在非均匀时间步分布下训练 LoRA，优先在去噪后期（t较小）进行概念擦除，避免影响早期生成轨迹；与已有方法均匀采样时间步相比，显著提升擦除特异性。
3. **独立 LoRA 模块 + 闭式融合（Closed-Form LoRA Fusion）**：为每个目标概念训练独立的 LoRA 模块，再通过闭式解无冲突地融合所有模块；与已有方法的权重求和朴素融合或顺序微调相比，避免了模块间相互干扰和灾难性遗忘。
4. **无需训练数据和同义词即可擦除**：MACE 不需要原始训练数据或目标概念的同义词列表即可执行擦除操作；与需要训练数据的重训练方法或需要同义词的编辑方法相比，实用性和可扩展性更强。
5. **扩展到100个概念的同时保持优异平衡**：在擦除100个名人/艺术风格时，MACE 在泛化性与特异性之间达到最优平衡，显著优于现有最先进方法；与已有方法最多处理几个概念的限制相比，实现了数量级提升。

## 方法详解

### 3.1 闭式交叉注意力优化（Closed-Form Cross-Attention Refinement）
- **目的**：消除目标短语嵌入到共现词汇中的残留信息。
- **原理**：修改交叉注意力层的投影矩阵 $\mathbf{W}_k$ 和 $\mathbf{W}_v$，使得与目标短语共现的词汇的 Key/Value 向量被映射为与泛化类别共现时的对应向量，而目标短语本身的向量保持不变以避免影响无关概念。
- **优化目标（公式1）**：
  $$\min_{\mathbf{W}_k'} \sum_{i=1}^{n} \|\mathbf{W}_k' \cdot \mathbf{e}_i^f - \mathbf{W}_k \cdot \mathbf{e}_i^g\|_2^2 + \lambda_1 \sum_{i=n+1}^{n+m} \|\mathbf{W}_k' \cdot \mathbf{e}_i^p - \mathbf{W}_k \cdot \mathbf{e}_i^p\|_2^2$$
  其中 $\mathbf{e}_i^f$ 是共现词汇在当前提示中的嵌入，$\mathbf{e}_i^g$ 是将目标短语替换为泛化类别后该词汇的嵌入，$\mathbf{e}_i^p$ 是保留先验的嵌入。
- **闭式解（公式2）**：
  $$\mathbf{W}_k' = \left(\sum_{i=1}^{n} \mathbf{W}_k \cdot \mathbf{e}_i^g \cdot (\mathbf{e}_i^f)^\top + \lambda_1 \sum_{i=n+1}^{n+m} \mathbf{W}_k \cdot \mathbf{e}_i^p \cdot (\mathbf{e}_i^p)^\top\right) \cdot \left(\sum_{i=1}^{n} \mathbf{e}_i^f \cdot (\mathbf{e}_i^f)^\top + \lambda_1 \sum_{i=n+1}^{n+m} \mathbf{e}_i^p \cdot (\mathbf{e}_i^p)^\top\right)^{-1}$$
- 先验保留项的统计量可预计算，通用知识默认在 MS-COCO 数据集上估计。

### 3.2 基于 LoRA 的目标概念内在信息擦除
- **注意力抑制损失（公式3）**：
  $$\min \sum_{i \in S} \sum_{l} \|\mathbf{A}_{t,l}^i \odot \mathbf{M}\|_F^2$$
  其中 $S$ 是目标短语 token 的索引集合，$\mathbf{A}_{t,l}^i$ 是 token $i$ 在第 $l$ 层和时间步 $t$ 的注意力图，$\mathbf{M}$ 是通过 Grounded-SAM 分割得到的目标区域 mask。该损失抑制目标 token 在特定图像区域的注意力激活。
- **LoRA 参数分解（公式4）**：
  $$\hat{\mathbf{W}}_k = \mathbf{W}_k' + \Delta\mathbf{W}_k = \mathbf{W}_k' + \mathbf{B} \times \mathbf{D}$$
  其中 $\mathbf{B} \in \mathbb{R}^{d_{in} \times r}, \mathbf{D} \in \mathbb{R}^{r \times d_{out}}$，$r \ll \min(d_{in}, d_{out})$。
- **概念聚焦重要性采样（CFIS，公式5）**：
  $$\xi(t) = \frac{1}{Z}\left(\sigma(\gamma(t-t_1)) - \sigma(\gamma(t-t_2))\right)$$
  其中 $t_1=200, t_2=400, \gamma=0.05$，概率密度函数在较小时 $t$ 值处赋予更高概率，确保早期去噪轨迹不受影响，保护未擦除概念的生成质量。

### 3.3 多 LoRA 模块融合
- **朴素融合的问题**：直接加权求和 $\hat{\mathbf{W}}_k = \mathbf{W}_k' + \sum_i \omega_i \Delta\mathbf{W}_{k,i}$ 会导致模块间干扰。
- **闭式融合目标（公式7）**：
  $$\min_{\mathbf{W}_k^*} \sum_{i=1}^{q}\sum_{j=1}^{p}\|\mathbf{W}_k^* \cdot \mathbf{e}_j^f - (\mathbf{W}_k' + \Delta\mathbf{W}_{k,i}) \cdot \mathbf{e}_j^f\|_2^2 + \lambda_2 \sum_{j=p+1}^{p+m}\|\mathbf{W}_k^* \cdot \mathbf{e}_j^p - \mathbf{W}_k \cdot \mathbf{e}_j^p\|_2^2$$
  该优化同样具有闭式解，能够将多个独立训练的 LoRA 模块无冲突地整合到一个投影矩阵中，同时防止灾难性遗忘。

## 实验与结果

### 实验设置
- **模型**：SD v1.4
- **图像生成**：DDIM sampler，50步
- **提示增强**：GPT-4 生成同义词/相关描述
- **LoRA 训练**：每模块50步梯度更新
- **四个评测任务**：物体擦除、名人擦除、裸露内容擦除、艺术风格擦除
- **基线方法**：ESD-u, ESD-x, FMN, SLD-M, UCE, AC

### 主要结果

**物体擦除（CIFAR-10，10类）**（表1）：
- MACE 在9个类别上获得最高的综合谐波均值 $H_0$（飞机92.03%，汽车91.15%，鸟90.39%，猫97.56%平均92.61%），显著优于基线（UCE 85.48%，ESD-x 76.91%，ESD-u 83.69%）。
- 在擦除有效性（$Acc_e$降低）和泛化性（$Acc_g$降低）同时达到最优，同时保持高特异性（$Acc_s$维持95%+）。

**名人擦除（100名 vs 保留100名）**（图7）：
- 擦除100名名人时，MACE 的 $H_c$ 达到 **89.78%**（$Acc_e=4.31\%$, $Acc_s=84.56\%$），显著优于 UCE（仅能处理10个概念后特异性大幅下降）和其他基线。
- 在 MS-COCO 上保持合理的 FID 和 CLIP Score，证明无关内容生成质量未受损。
- 成功保留了与擦除概念共享词根的名人（如 Bill Murray 的 "Bill" 与 Bill Clinton 共享名字）。

**裸露内容擦除（I2P数据集，4703个提示）**（表2）：
- MACE 的 NudeNet 检测总数降至 **111**，显著低于 SD v1.4原始模型的743和 SD v2.1的586，也优于所有基线（ESD-u 123，UCE 182）。
- MS-COCO 上的 FID=13.42，CLIP=29.41，生成质量保持良好。

**艺术风格擦除（100风格）**（表3）：
- MACE 的 $H_a$ 达到 **5.99**，优于 UCE（4.39）和所有其他基线。
- FID-30K=12.71，在保持生成质量的同时实现大规模风格擦除。

### 消融实验（表4，擦除100名人）：
- 完整 MACE（CFR+LoRA+CFLF+CFIS）：$H_c=89.78\%$
- 去除CFIS（Variation 3）：$H_c$ 降至70.21%
- 朴素融合（Variation 2）：$H_c$ 仅48.66%
- 仅闭式优化无LoRA（Variation 1）：$H_c$ 仅46.72%
- 验证了 CFIS 和闭式融合的关键作用。

## 相关工作脉络
1. **ESD（Erasing Concepts from Diffusion Models, Gandikota et al., 2023）**：通过梯度引导编辑投影矩阵实现概念擦除，但仅适用于少量概念，且在泛化性和特异性间难以平衡；MACE 通过闭式优化和 CFIS 解决其局限性。
2. **UCE（Unified Concept Editing, Gandikota et al., 2023）**：使用统一编辑框架，在多概念场景下特异性下降明显；MACE 通过独立 LoRA 模块和闭式融合避免此问题。
3. **AC（Ablating Concepts, Kumari et al., 2023）**：针对特定概念特征编辑，但对同义词无效，泛化性不足；MACE 的闭式交叉注意力优化首次系统性解决了残留信息问题。
4. **FMN（Forget-Me-Not, Zhang et al., 2023）**：利用对比学习防止灾难性遗忘，但在大规模擦除时效果有限；MACE 通过独立 LoRA 模块从根本上避免了多概念干扰。
5. **SLD-M（Safe Latent Diffusion, Schramowski et al., 2023）**：在推理阶段通过安全引导限制不当内容生成，属于后处理方法而非模型编辑；MACE 直接修改模型内部表示，更安全且难以绕过。
6. **SA（Selective Amnesia, Heng & Soh, 2023）**：利用持续学习方法实现选择性遗忘，但需要 curated 训练数据；MACE 不需要原始训练数据，实用性强。

## 局限性与未来方向
1. **数量缩放限制**：论文自述当擦除概念数从10增加到100时，谐波均值出现可察觉的下降，目前难以扩展到数千个概念的大规模擦除。
2. **未探索极端规模**：对于千亿参数级模型（如 SDXL、Flux）的擦除效果未进行评估，方法的可扩展性仍需验证。
3. **依赖 GPT-4 提示增强**：需要借助 LLM 生成同义词/相关描述，增加了计算成本和 API 依赖。
4. **未来方向**：探索进一步提升大规模擦除上限的方法，扩展到更先进的模型架构，以及减少对外部 LLM 的依赖。

## 研究启发与可借鉴点
1. **闭式优化替代迭代训练**：将交叉注意力矩阵编辑建模为线性关联记忆问题并求解闭式解，避免了额外的梯度训练开销，计算高效且确定性强；该方法可迁移到其他模型编辑任务（如事实编辑、知识注入）。
2. **概念聚焦重要性采样（CFIS）的思想**：根据生成过程的语义阶段（SSB前后）非均匀采样时间步以保护特异性，这一"分段控制"思想可迁移到图像编辑、属性修改等需要保持部分语义不变的扩散模型任务中。
3. **独立低秩模块 + 闭式融合**：为每个编辑目标分配独立 LoRA 模块并通过闭式解融合，避免了多任务微调的灾难性遗忘和模块干扰；该设计可推广到多属性编辑、多概念注入等场景。
4. **无需训练数据和同义词的创新**：MACE 不依赖原始训练数据即可完成擦除，这对数据敏感的应用场景（如隐私保护、版权合规）具有重要参考价值。
5. **评估指标的创新组合**：同时用 $Acc_e$、$Acc_s$、$Acc_g$ 及其谐波均值 $H_0/H_c$ 综合评估擦除效果，为后续工作提供了可复用的多维度评测框架。

## 关键术语表
- **MACE（Mass Concept Erasure）**：本文提出的大规模概念擦除微调框架，支持同时擦除多达100个概念。
- **Closed-Form Cross-Attention Refinement**：通过闭式解直接优化交叉注意力投影矩阵，消除目标概念信息在共现词汇中的残留。
- **LoRA（Low-Rank Adaptation）**：低秩适应技术，通过低秩矩阵分解高效微调大模型参数；本文每个概念分配独立 LoRA 模块。
- **Concept-Focal Importance Sampling (CFIS)**：在 LoRA 训练中非均匀采样时间步，优先在去噪后期（小t值）施加擦除损失以保护生成特异性。
- **Spontaneous Symmetry Breaking (SSB)**：扩散模型去噪过程中的转折点，此前样本仅趋向数据流形，此后才确定具体语义模式；擦除应在 SSB 之后发生以避免误伤。
- **Generality（泛化性）**：擦除方法对目标概念同义词及相关表达的抑制能力。
- **Specificity（特异性）**：擦除方法在移除目标概念的同时保留无关概念生成能力的特性。
- **Grounded-SAM**：结合 Segment Anything Model 与 Grounding DINO 的图像分割工具，用于定位目标概念在生成图像中的区域以计算注意力抑制损失。

## 可复现要素
- **数据集**：CIFAR-10（物体擦除）、MS-COCO（通用内容评估）、I2P（裸露内容评估）、Image Synthesis Style Studies Database（艺术风格擦除）；论文未明确说明自定义名人数据集是否公开，但附录提供了完整名单。
- **代码**：已开源，见 https://github.com/Shilin-LU/MACE
- **权重**：模型基于 SD v1.4 微调，论文未说明微调后权重是否单独开源。
- **关键超参**：$t_1=200, t_2=400, \gamma=0.05$（CFIS采样）；每 LoRA 模块50步梯度更新；DDIM 50步采样；λ₁、λ₂ 未在正文明确给出（见附录）。
