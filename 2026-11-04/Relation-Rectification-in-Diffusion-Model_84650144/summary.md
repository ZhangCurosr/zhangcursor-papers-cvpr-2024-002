---
title: "Relation-Rectification-in-Diffusion-Model"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Wu_Relation_Rectification_in_Diffusion_Model_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 08:33:26"
field: "文本到图像生成与关系理解"
keywords: ["Diffusion Model", "Relation Generation", "Text-to-Image", "Graph Convolutional Network", "Vision-Language Model", "Compositional Generation"]
innovations: ["提出Relation Rectification任务与RRNet框架，通过HGCN修正[EOT]嵌入提升方向性关系生成准确率", "发现SD中[EOT]嵌入对关系控制的关键作用且OSP对的[EOT]嵌入高度相似", "设计对象解耦与正负双重损失机制，在冻结预训练模型前提下仅需20分钟微调"]
benchmarks: ["Relation Rectification Benchmark (RR Dataset)", "COCO Validation Set (FID计算)"]
---

# 论文速读：Relation-Rectification-in-Diffusion-Model

## 一句话总结
本文提出**Relation Rectification**新任务与**RRNet**框架，通过异构图卷积网络（HGCN）修正文本编码器中承载关系语义的`[EOT]`嵌入，使Stable Diffusion能准确生成具有方向性关系（如"碗在布里面" vs "布在碗里面"）的图像，同时保持预训练模型参数冻结。

## 研究问题与动机
1. **关系生成缺陷**：大型T2I扩散模型在生成含方向性关系的图像时经常出错，如提示"a book is placed on a bowl"易被误生成为"a bowl is placed on a book"。
2. **VLM的Bag-of-Words局限**：CLIP等视觉语言模型采用对比学习训练，倾向于将句子理解为"词袋"而忽略组合语义和对象逻辑顺序，导致其生成的文本嵌入无法区分关系方向。
3. **现有方法未触及根本**：Canvas布局引导等方法仅规避问题而非修复文本编码器对关系方向性的表征缺陷。
4. **关键发现**：OSP（对象交换提示）对的`[EOT]`嵌入向量余弦相似度接近1，几乎不可区分，这是导致关系生成失败的根源。

## 核心贡献（创新点）
1. **提出Relation Rectification新任务**：定义面向方向性关系生成的评估基准，区分于现有的简单空间关系研究。
2. **揭示[EOT]嵌入的关键作用**：发现SD严重依赖`[EOT]`特殊标记的嵌入来控制关系生成，且该嵌入在OSP对中高度相似。
3. **设计RRNet+HGCN架构**：首次将异构图卷积网络引入文本嵌入修正，显式建模关系方向性图结构，仅需训练轻量HGCN而冻结SD与文本编码器。
4. **构建Relation Rectification Benchmark**：包含21种关系（8种位置+13种动作）、4200张图像的评测数据集，填补该领域评估空白。

## 方法详解
**图结构建模**：将一对OSP（如`<A,R,B>`和`<B,R,A>`）建模为有向异构图，包含三类节点：对象节点$v_O$、关系节点$v_R$、调整节点$v_{\Delta EOT}$，其中对象与关系节点使用CLIP词嵌入初始化。

**HGCN消息传递**：通过异构图卷积网络沿边聚合信息，更新公式为：
$$h_{\Delta EOT}^{(l+1)} = \sum_{\varepsilon \in \hat{\mathcal{E}}} \sigma\left(b^{(l)} + \sum_{j:(e_{j,i}) \in \mathcal{E}} \alpha_{j,i} h_j^{(l)} W_\varepsilon^{(l)}\right)$$
其中不同边类型使用独立权重矩阵$W_\varepsilon$。

**嵌入修正**：调整后`[EOT]`嵌入通过加法更新：$V'_{eot} = V_{eot} + \lambda \cdot h_{\Delta EOT}^{(L)}$，$\lambda \in [0,1]$控制修正强度。

**对象节点解耦**：为每个对象节点分配独立的$v_{\Delta EOT}^A$，从模板句"This is a photo of {A}"提取$V_{eot}^A$并添加高斯噪声，迫使学习有效调整向量而非平凡解。

**双重损失**：
- **正损失（去噪损失）**：$\mathcal{L}_{denoise} = \mathbb{E}[||\epsilon - \epsilon_\theta(x_t, t, \phi(c(y)))||_2^2]$，使调整后的嵌入与参考图像对齐
- **负损失**：$\mathcal{L}_{neg} = \mathbb{E}[-||\epsilon - \epsilon_\theta(x_t, t, \phi(c(\tilde{y})))||_2^2]$，最大化OSP对的`[EOT]`嵌入距离
- 总损失：$\mathcal{L} = \eta \cdot \mathcal{L}_{denoise} + \xi \cdot \mathcal{L}_{neg}$，论文取$\eta=10, \xi=2$

## 实验与结果
**数据集**：RR Dataset包含21种关系（8位置+13动作），每提示生成100张图像，共4200张；训练在SD 2-1上进行100 epochs，约20分钟/V100。

**评估指标**：
- **关系生成准确率**：使用Qwen-VL-Chat与LLaVA作为视觉语言评估器，判断生成图像是否符合提示关系
- **FID**：衡量图像质量，与COCO val计算

**主要结果**（Table 1）：
| 方法 | Position(Qwen)↑ | Action(Qwen)↑ | OGA↑ | FID↓ |
|------|----------------|---------------|------|------|
| SD | 0.467 | 0.399 | 0.898 | 83.73 |
| RRNet (λ=1) | **0.729** | **0.490** | **0.971** | 110.01 |
| RRNet (λ=0.6) | 0.697 | 0.500 | 0.970 | 100.78 |

- 最佳关系准确率较SD提升约**25-26个百分点**
- λ在0.4-0.6时能平衡准确率与图像质量

**消融实验**（Table 2）：移除HGCN、负损失、对象解耦均导致性能下降，验证三者必要性。

**用户研究**（Table 3）：63名评估者中RRNet偏好率达**75.56%**，较SD（16.98%）提升58.58个百分点。

**零样本泛化**：在未见对象组合上仍能生成正确关系方向（Figure 1(b)）。

## 相关工作脉络
1. **Personalized Diffusion**（Textual Inversion, DreamBooth等）：优化CLIP文本编码器生成个性化概念，但会改变对象语义；本文聚焦关系而非对象属性，且保持嵌入空间不变。
2. **Compositional Image Generation**（ControlNet, GLIGEN等）：提供额外条件（边界框、深度图）控制生成；本文不引入外部条件，仅修正文本嵌入内部关系表征。
3. **Training-free Layout Control**（Attend-and-Excite, BoxDiff等）：通过修改cross-attention布局控制对象位置；本文方法需少量参考图像训练但泛化性更强。
4. **Vision-Language Models**（CLIP等）：受Bag-of-Words问题影响；本文识别其对T2I的关系生成缺陷并提出针对性修正。
5. **Graph-based VLMs**（Scene Graphs, Structure-CLIP等）：利用图结构增强VLM；本文首次将HGCN用于扩散模型文本嵌入修正，且训练效率极高。

## 局限性与未来方向
1. **图像质量-关系准确率的权衡**：较大λ值（如1.0）导致FID显著升高（83.73→110.01），需人工调节。
2. **依赖参考图像**：训练需要每个关系类型的少量参考图像，限制了完全无监督的应用场景。
3. **复杂关系扩展性未知**：当前聚焦简单二元关系（A R B），对多对象、嵌套关系的处理能力未验证。
4. **基座模型限制**：仅在SD 2-1上验证，对SDXL、SD3等新一代模型的泛化性待探索。

## 研究启发与可借鉴点
1. **图结构建模文本语义**：将句子关系建模为异构图并利用HGCN聚合信息，是提升组合生成能力的有效范式，可迁移至多模态对齐、具身导航等任务。
2. **冻结大模型+轻量适配器**：仅训练HGCN而冻结SD与文本编码器，参数高效且避免灾难性遗忘，适合工业部署。
3. **正负损失配对训练**：OSP对的正负去噪损失设计可推广至其他需要区分相似语义对比的学习场景（如细粒度分类、锚点学习）。
4. **特殊标记嵌入分析**：通过掩码实验定位关键语义位置（[EOT]），再针对性修正的思路可复用至LLM/VLM的可解释性分析与编辑。
5. **客观评估构建**：使用Qwen-VL-Chat/LLaVA作为自动评估器的pipeline，为关系生成领域提供了可复用的评测标准。

## 关键术语表
- **Relation Rectification**：关系修正任务，指让扩散模型准确生成符合文本方向性关系描述的图像
- **OSP（Object-Swapped Prompts）**：对象交换提示，描述相同关系但对象位置互换的提示对（如"A在B里面"vs"B在A里面"）
- **[EOT]**：End of Text特殊标记，其嵌入向量在对比学习中积累整个句子的组合语义，是关系生成的关键
- **HGCN（Heterogeneous Graph Convolutional Network）**：异构图卷积网络，通过不同权重函数处理不同类型节点和边的图神经网络
- **Object Node Disentanglement**：对象节点解耦，防止对象语义与关系语义在图中混合的机制
- **Negative Loss**：负损失，通过最大化OSP对嵌入距离来强制区分关系方向的对比损失

## 可复现要素
- **代码**：已开源，项目页面https://wuyinweihah.github.io/rrnet.github.io/
- **数据集**：RR Dataset（自行构建，4200张图像，论文未公开原始数据；代码含数据生成脚本）
- **基座模型**：Stable Diffusion 2-1（Hugging Face可下载）
- **关键超参**：训练epoch=100，λ=1.0（训练），生成时λ∈[0,1]可调，η=10，ξ=2
- **推理设置**：PNDM调度器，30步去噪，classifier-free guidance
- **硬件**：单卡NVIDIA V100，训练约20分钟/关系
