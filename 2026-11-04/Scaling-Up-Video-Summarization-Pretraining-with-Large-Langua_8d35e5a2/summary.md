---
title: "Scaling-Up-Video-Summarization-Pretraining-with-Large-Langua"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Argaw_Scaling_Up_Video_Summarization_Pretraining_with_Large_Language_Models_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 16:38:06"
field: "视频理解与摘要"
keywords: ["video summarization", "large language models", "self-supervised pretraining", "multimodal learning", "autoregressive generation"]
innovations: ["利用LLM作为Oracle summarizer自动生成大规模视频摘要预训练数据集（25万对）", "将视频摘要从并行分类转化为自回归连续特征解码，缓解长尾分布与重复片段问题", "提出支持有/无旁白视频的统一多模态自回归框架，并在LfVS-T基准上建立新SOTA"]
benchmarks: ["LfVS-T", "TVSum", "SumMe"]
---

# 论文速读：Scaling Up Video Summarization Pretraining with Large Language Models

## 一句话总结
论文提出利用大语言模型（LLMs）作为Oracle summarizers，通过自动化的语音转录-提取式摘要生成-视频片段映射流水线，构建包含25万视频-摘要对的大规模预训练数据集（LfVS-P），并提出一种基于自回归解码的多模态Transformer网络，在多个视频摘要基准上达到新的state-of-the-art。

## 研究问题与动机
- **数据集规模瓶颈**：现有视频摘要数据集（TVSum仅50个、SumMe仅25个）规模极小，导致SOTA模型在跨域泛化时严重过拟合特定领域。
- **长尾分布问题**：现有方法将视频摘要建模为逐帧/逐片段的二元分类或重要性分数预测，摘要片段占比极低，类别严重不平衡。
- **独立预测导致的重复**：现有方法并行预测各时间步的重要性，忽略摘要片段之间的时序依赖关系，导致大量重复片段被选入摘要。
- **多模态利用不足**：尽管语音-视频对齐信息丰富，但现有工作对文本模态的利用有限，且难以处理无旁白的视频场景。

## 核心贡献（创新点）
1. **LLM驱动的自动化数据集构建流水线**：利用HowTo100M中的旁白视频、Whisper语音转录与LLM提取式摘要生成pseudo-ground truth，构建25万规模的LfVS-P数据集，实现了从文本摘要到视频摘要的无损时间映射。
2. **自回归回归式视频摘要模型**：将视频摘要从离散的二分类/重要性分数预测转化为连续特征的自回归解码，通过前序已选摘要片段条件化后续生成，从根本上缓解长尾分布和重复片段问题。
3. **支持有/无文本输入的统一多模态框架**：训练时随机以MASK token掩盖文本输入（0%~100%掩码率），使模型在推理时既能利用旁白文本也能纯视觉处理无旁白视频。
4. **提出LfVS-T新基准**：包含1,200个8~33分钟长视频、392个类别、专业人工标注的高质量摘要，填补了长视频摘要评估基准的空白。

## 方法详解

### 数据集构建流水线
- **源数据筛选**：从HowTo100M（1.2M+旁白视频）中选取≥8分钟视频，用Whisper转录为文本；通过CLIP相似度过滤语音-视觉对齐不足的样本。
- **LLM提取式摘要生成**：将带时间戳的句子列表输入LLM（GPT-3.5-16K为主，辅以GPT-4、Llama-2-13B对比），提示模型选择不重复的原句并保持原始措辞和时间戳，确保可逆映射。
- **伪地面真实生成**：利用时间戳定位对应视频片段，再用CLIP最近邻搜索修正可能的时间戳偏差，聚合得到pseudo-ground truth摘要视频序列。

### 模型架构（Encoder-Decoder Transformer）
- **视频编码**：CLIP-ViT-L/14提取每帧视觉token，拼接SOS/EOS后送入6层Transformer V-Encoder（隐藏维度1024，8头注意力）学习时序上下文：
  $$\{\hat{x}_i\} = \text{V-Encoder}(\{\text{SOS}, x_1, \dots, x_n, \text{EOS}\})$$
- **文本编码**：SRoBERTa-NLI-large提取句子嵌入，经3层T-Encoder得到上下文化文本表示：
  $$\{\hat{s}_j\} = \text{T-Encoder}(\text{SRoBERTa}(\{S_k\}))$$
- **跨模态注意力**：以视觉特征为Query、文本特征为Key/Value的multi-head cross-attention融合双模态信息：
  $$\text{head} = \text{Attn}(\hat{x}W^Q, \hat{s}W^K, \hat{s}W^V)$$
- **自回归摘要解码**：6层Transformer Summary Decoder逐时刻生成摘要片段的CLIP特征表示，每一步依赖前序已生成片段（teacher forcing训练，SOS起始自回归推理）：
  $$\hat{y}_t = \text{S-Decoder}(\{\hat{x}_i^{\hat{s}}\}, \{\text{SOS}, y_1, \dots, y_{t-1}\})$$
- **训练损失**：预测摘要特征与pGT摘要特征的MSE损失：$\mathcal{L} = \sum_{i=1}^{m+1} |\hat{y}_i - y_i|^2$。
- **推理**：用nearest neighbor检索将解码特征匹配到输入视频帧，聚合得到摘要视频。

## 实验与结果

### 数据集与评估
- **LfVS-P**：250K视频-摘要对，平均时长13.3分钟，用于预训练。
- **LfVS-T**：1,200个长视频，人工标注，用于测试评估。
- **传统基准**：TVSum（50视频）、SumMe（25视频），使用F1 Score、Kendall's τ、Spearman's ρ三种指标。

### LfVS-T基准对比（Table 2）
| 方法 | F1 Score | τ | ρ |
|------|----------|---|---|
| CLIP-It | 62.87 | 0.129 | 0.225 |
| TL:DW? | 66.25 | 0.138 | 0.233 |
| iPTNet | 65.80 | 0.140 | 0.237 |
| A2Summ | 66.04 | 0.143 | 0.246 |
| **Ours** | **68.11** | **0.158** | **0.277** |

本文较第二优方法TL:DW?在F1上提升**1.86%**，较A2Summ提升**2.07%**。

### SumMe/TVSum对比与跨数据集泛化（Table 3）
- **直接训练**：在两个基准上均达到SOTA（SumMe F1=56.94，TVSum F1=66.04；τ/ρ指标优势更明显）。
- **LfVS-P预训练→SumMe/TVSum fine-tune**：F1分别达到**60.42**和**72.38**，较从头训练分别提升**6.1%**和**9.1%**，验证大规模预训练的泛化收益。

### 消融分析（Table 4-5）
- **去掉文本输入**：F1下降2.3%（66.59→68.11），证明多模态训练的必要性。
- **去掉V-Encoder/T-Encoder/Cross-Attention**：各自造成1~3%的F1下降。
- **分类 vs 自回归**：分类 baseline F1=63.31，自回归提升**4.8%**，验证序列建模的重要性。
- **数据集规模**：10%/50%/100%预训练数据对应F1为53.44/64.58/68.11，规模效应显著。
- **LLM选择**：GPT-4生成数据（55.96）> GPT-3.5（53.44）> Llama-2-13B（44.89），模型能力直接影响pseudo-GT质量。

## 相关工作脉络
1. **CLIP-It [25]**：语言引导的视频摘要，但依赖预定义query；本文无需query，通过旁白文本自动指导摘要生成，且支持无文本场景。
2. **TL:DW? [26]**：教学视频摘要的多模态方法，使用cross-modal saliency；本文扩展其思想至任意长视频，并首次提出自回归解码范式。
3. **A2Summ [8]**：双对比损失对齐视频-文本摘要；本文不依赖对比损失，而是通过自回归特征回归直接生成摘要。
4. **iPTNet [10]**：纯视觉的联合摘要与时刻定位；本文利用文本模态并证明多模态训练提升显著，同时解决其独立的并行预测问题。
5. **TextRank / BART / Pegasus**：传统与LLM驱动文本摘要方法；本文将LLM的提取式摘要能力迁移至视频域，通过时间戳保持与CLIP对齐实现模态转换。

## 局限性与未来方向
- **伪标签噪声**：自动生成的pGT依赖LLM输出质量和Whisper转录精度，可能存在错误摘要片段，尤其当LLM未能严格遵循prompt指令时（如Llama-2-13B实验所示）。
- **长视频计算开销**：1fps采样+CLIP编码+Transformer编码对超长视频仍存在显存和推理延迟压力。
- **无旁白视频性能下降**：虽然支持MASK文本输入，但ablation显示去除文本后F1下降2.3%，纯视觉模式仍有较大提升空间。
- **评估偏向长视频**：LfVS-T为长视频设计，在短视频场景（如SumMe/TVSum）上的优势相对有限。

## 研究启发与可借鉴点
1. **LLM作为数据合成Oracle**：用强LLM生成伪标签、再通过可逆映射还原到目标模态的思路，可迁移至图像描述、动作识别、视频描述等其他弱监督数据稀缺任务。
2. **自回归解码替代并行分类**：将"逐帧分类/分数预测"范式改为"逐片段自回归生成"，能有效缓解类别不平衡和重复选取问题，适用于各类时序序列生成任务（如关键帧选取、视频剪辑）。
3. **随机模态掩码提升鲁棒性**：训练时随机屏蔽某一模态（文本/视觉）的引入方式，可使模型在推理时灵活适配有无该模态的场景，适合多模态通用模型的预训练策略设计。
4. **CLIP最近邻修正时间戳偏差**：在文本→视频映射阶段用CLIP嵌入做局部精修，是一种低成本提升对齐精度的通用技巧。

## 关键术语表
**LFVS-P**：Long-form Video Summarization Pretraining数据集，25万条自动生成的视频-摘要对，用于大规模预训练。
**LFVS-T**：Long-form Video Summarization Testing基准，1,200个长视频+人工标注摘要，用于评估视频摘要模型性能。
**Oracle Summarizer**：指利用强LLM作为"理想摘要器"生成高质量伪标签，以替代昂贵的人工标注。
**Extractive Summarization**：提取式摘要，从原文中选取重要片段保留原词而非 paraphrase，本文用于保证时间戳可逆映射。
**Self-attention Video Encoder**：基于Transformer的时序编码模块，使视频各帧互相 attend 以学习上下文表征。
**Cross-modal Attention**：以视频特征为Query、文本特征为Key/Value的注意力机制，融合视觉与语言线索。
**Autoregressive Decoding**：自回归解码，每一步生成依赖前序已生成片段，捕捉摘要片段间的序列依赖。
**Pseudo-Ground Truth (pGT)**：伪地面真实，由LLM自动生成并经时间映射得到的摘要视频，作为训练监督信号。

## 可复现要素
- **数据集**：LfVS-P由HowTo100M（公开）加工生成，具体处理脚本见论文附录；LfVS-T包含1,200个YouTube视频，论文未明确声明开源状态，但提及"publicly available"。
- **代码**：论文未提供官方开源链接（标注为"official implementations"链接需另行查找）。
- **权重**：未公开预训练权重。
- **关键超参**：采样率1fps；CLIP-ViT-L/14；SRoBERTa-NLI-large；V-Encoder 6层、T-Encoder 3层、S-Decoder 6层，hidden dim=1024，8 attention heads，FFN dim=2048；AdamW，初LR=3e-4，cosine annealing；batch size=64，100 epochs，4×A6000 GPU。
- **LLM**：主实验使用GPT-3.5-16K，对比实验使用GPT-4和Llama-2-13B。
- **语音转录**：Whisper（开源）。
