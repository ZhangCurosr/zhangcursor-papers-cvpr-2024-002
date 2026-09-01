---
title: "Relation-Rectification-in-Diffusion-Model"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Wu_Relation_Rectification_in_Diffusion_Model_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 15:21:16"
field: "文本到图像生成"
keywords: ["文本到图像生成", "扩散模型", "关系建模", "异构图卷积网络", "视觉-语言对齐"]
innovations: ["提出Relation Rectification任务和RRNet框架，通过HGCN调整[EOT]嵌入解决方向性关系生成问题", "发现[EOT]嵌入对关系生成的关键作用及OSPs嵌入不可区分的问题", "设计正负损失联合优化策略，仅需训练轻量级HGCN而保持SD参数冻结"]
benchmarks: ["Relation Rectification Benchmark", "COCO"]
---

# 论文速读：Relation-Rectification-in-Diffusion-Model

## 一句话总结
论文针对预训练文本到图像扩散模型在处理含方向性关系提示词时易混淆物体间关系的问题，提出了Relation Rectification任务及RRNet方法，通过异构图卷积网络微调[EOT]嵌入向量，使模型能够准确生成符合文本关系描述的图像。

## 研究问题与动机
- 大规模T2I扩散模型（如SD）在处理包含方向性关系的提示词时，经常无法准确呈现物体间的视觉关系（如将"书放在碗上"生成为"碗放在书上"）。
- 现有VLMs（如CLIP）的对比学习目标偏向全局语义，未能显式学习句子顺序信息，导致模型倾向于将文本理解为"词袋"，无法捕捉组合语义。
- 现有方法（如Canvas布局引导、ControlNet等）仅通过外部干预绕过问题，未从根本上解决文本编码器对关系方向性不敏感的核心缺陷。
- 研究发现[EOT]特殊标记的嵌入对控制关系生成起关键作用，但来自对象交换提示对（OSPs）的[EOT]嵌入余弦相似度接近1，几乎不可区分。

## 核心贡献（创新点）
- **提出Relation Rectification新任务**：定义了让模型对OSPs给出不同响应的任务目标，与前人将提示简化为"词袋"的做法形成本质区别。
- **发现[EOT]嵌入是关系生成的关键瓶颈**：通过消融实验证明[EOT]嵌入被遮挡会严重破坏关系生成，而现有工作未系统研究此问题。
- **设计基于HGCN的RRNet框架**：将OSPs建模为有向异构图，通过图卷积生成调整向量来分离OSPs的[EOT]嵌入，无需修改SD和文本编码器参数。
- **构建Relation Rectification Benchmark**：包含21种关系（8种位置+13种动作）、4200张图像的综合评测数据集。
- **提出正负损失结合的联合优化策略**：通过L_denlose和L_neg的组合，在提升关系生成准确性的同时防止语义混淆。

## 方法详解
**RRNet架构**：
- 将一对OSPs（如<A, R, B>和<B, R, A>）建模为有向异构图，包含三类节点：$V_O$（对象节点）、$V_R$（关系节点）、$V_{\Delta EOT}$（用于学习调整向量的节点）。
- 对象节点和关系节点的初始表示采用CLIP词嵌入，$V_{\Delta EOT}$随机初始化。
- 通过HGCN进行多轮图卷积，不同类型节点采用不同的权重矩阵$W_\varepsilon$进行信息聚合：
$$h_{\Delta EOT}^{(l+1)} = \sum_{\varepsilon \in \hat{\mathcal{E}}} \sigma(b^{(l)} + \sum_{j:(e_{j,i})\in\mathcal{E}} \alpha_{j,i} h_j^{(l)} W_\varepsilon^{(l)})$$
- 调整后的[EOT]嵌入为：$V_{eot}' = V_{eot} + \lambda \cdot h_{\Delta EOT}^{(L)}$，其中$\lambda \in [0,1]$控制调整强度。

**对象节点解耦机制**：
- 为避免物体特征与关系特征混合，引入专用节点$v_{\Delta EOT}^A$用于解耦物体A，通过模板句"This is a photo of {A}"提取其$V_{eot}^A$并添加高斯噪声，迫使模型学习有效调整向量而非平凡解。

**损失函数设计**：
- 正向损失（L_denlose）：基于参考图像的学习信号，引导RRNet生成与图像关系语义一致的调整向量。
- 负向损失（L_neg）：针对OSPs的反向版本，确保两者的[EOT]嵌入充分分离。
- 总损失：$\mathcal{L} = \eta \cdot \mathcal{L}_{denoise} + \xi \cdot \mathcal{L}_{neg}$，其中$\eta=10$，$\xi=2$。

## 实验与结果
**数据集与评估**：
- RR Dataset包含21种关系（8种位置关系如below/on the left，13种动作关系如touch/follow），每种生成100张图像，共4200张用于评测。
- 评估指标：基于Qwen-VL-Chat和LLaVA的关系生成准确率、目标生成准确率（OGA）、FID。

**主要结果**：
- 位置关系（Qwen）：RRNet(λ=1)达到0.729，相比SD基线的0.467提升约25%。
- 动作关系（Qwen）：RRNet达到0.490，相比SD基线的0.399提升约23%。
- OGA：RRNet(λ=1)达到0.971，保持较高的物体生成质量。
- FID：λ=0.6时FID=100.78，平衡了关系准确性与图像质量。
- 用户研究（63名评估者）：RRNet获得75.56%偏好率，vs SD的16.98%和Personalized Diffusion的7.46%。

**泛化能力**：能在未见对象上实现零样本关系生成（图6、图1b）。

## 相关工作脉络
- **与Personalized Diffusion对比**：后者通过微调CLIP文本编码器实现个性化，但会改变物体语义；RRNet仅训练轻量HGCN，保持文本编码器和SD参数冻结，只调整关系方向。
- **与Compositional Generation（GLIGEN、ControlNet等）对比**：这些方法依赖额外条件（如边界框、布局图）引导生成；RRNet从文本嵌入层面解决问题，无需额外条件输入。
- **与Attention-based方法（Attend-and-excite、BoxDiff）对比**：这些训练友好方法仅控制简单空间关系；RRNet能处理更复杂的动作关系和方向性关系。
- **与图增强的VLMs（Graph-CLIP、Structure-CLIP）对比**：这些方法用场景图增强VLM表示；RRNet将HGCN直接集成到扩散模型文本编码流程中。
- **与Bag-of-Words问题（Yuksekgonul et al. 2022）定位**：该工作揭示了VLM的"词袋"行为缺陷；RRNet在此基础上提出具体的嵌入空间修正方案。

## 局限性与未来方向
- 每种关系类型需单独训练RRNet，计算效率有待提升。
- 调整强度λ需要手动调参，较大的λ可能导致图像质量下降（FID升高）。
- 当前数据集仅涵盖21种关系，对复杂多对象场景的处理能力有限。
- 依赖少量参考图像进行训练，完全零样本场景的适用性需进一步验证。

## 研究启发与可借鉴点
- **[EOT]嵌入分析思路**：通过遮挡特殊标记的消融实验定位生成瓶颈，可迁移至其他视觉-语言模型的诊断分析。
- **异构图建模关系方向性**：HGCN对节点类型的区分处理（对象/关系/调整节点）为序列关系建模提供了新范式。
- **正负样本对比学习策略**：L_neg的设计思想可用于其他需要区分相似语义的任务（如反义关系、因果顺序）。
- **轻量级适配器设计**：仅训练附加模块而冻结主干，既保持基础模型能力又实现专项增强，适合资源受限场景。
- **对象解耦机制**：通过模板句和噪声注入防止特征混叠，可推广至多对象生成任务的语义保持。

## 关键术语表
- **[EOT]**：文本嵌入中的结束标记（End Of Text），其向量编码了整句语义，对控制生成图像的关系结构起关键作用。
- **OSP（Object-Swapped Prompts）**：对象交换提示对，由相同词语但顺序相反的提示组成，用于构建关系方向性对比。
- **HGCN（Heterogeneous Graph Convolutional Network）**：异构图卷积网络，通过不同类型节点和边的聚合函数处理异构图数据。
- **RRNet**：Relation Rectification Network的缩写，本文提出的基于HGCN的关系矫正网络框架。
- **Relation Rectification**：本文定义的新任务，指修正扩散模型对方向性关系的生成偏差。
- **OGA（Object Generation Accuracy）**：目标生成准确率，评估生成图像中实体是否被正确生成。
- **FID（Fréchet Inception Distance）**：通过预训练Inception模型衡量生成图像与真实图像的分布相似度。
- **$\lambda$**：调整强度超参数，控制$h_{\Delta EOT}$对$V_{eot}$的修改幅度，用于权衡关系准确性与图像质量。

## 可复现要素
- **数据集**：RR Dataset（21种关系，4200张图像），论文声明开源地址为https://wuyinweihah.github.io/rrnet.github.io/
- **代码/权重**：项目页面已公布，代码和预训练权重应已开源
- **关键超参**：λ∈[0,1]（生成时调整强度），η=10（正向损失权重），ξ=2（负向损失权重），训练100 epoch，1 NVIDIA V100 GPU约20分钟/关系，PNDM scheduler 30步，classifier-free guidance
