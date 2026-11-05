---
title: "Restoration-by-Generation-with-Constrained-Priors"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Ding_Restoration_by_Generation_with_Constrained_Priors_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 08:33:36"
field: "低级别视觉-图像恢复"
keywords: ["图像恢复", "扩散模型", "盲人脸恢复", "生成模型", "个性化恢复"]
innovations: ["提出噪声对齐的恢复即生成范式，无需假设退化类型", "通过生成专辑/个人专辑约束扩散模型生成空间以保持身份", "引入Skip Guidance周期性引导策略生成高质量anchor图像"]
benchmarks: ["Wider-Test", "LFW-Test", "WebPhoto-Test", "Deblur-Test"]
---

# 论文速读：Restoration-by-Generation-with-Constrained-Priors

## 一句话总结
论文提出一种基于预训练扩散模型的图像恢复方法，核心思想是对退化图像添加噪声后直接利用扩散模型去噪恢复，并通过fine-tune生成专辑或个人专辑来约束生成空间以保持输入身份特征，在无需假设退化类型的情况下实现了优秀的盲人脸恢复性能。

## 研究问题与动机
1. **退化未知的盲图像恢复难题**：现有判别学习方法依赖合成退化数据训练，泛化到真实复杂退化（如运动模糊、非均匀退化）时效果差；而基于模型的方法通常需要已知退化过程，缺乏实用性。
2. **扩散模型直接用于恢复的挑战**：预训练扩散模型主要在干净图像上训练，如何将退化图像"投影"到生成过程并在恢复质量与保留输入信息之间取得平衡是关键难点。
3. **身份保持与高频细节恢复的矛盾**：单次图像恢复是病态问题，需要额外约束以避免生成结果偏离输入身份；而已有参考图像方法（如ASFFNet）虽能保持身份但生成质量不足。

## 核心贡献（创新点）
1. **提出"恢复即生成"范式**：发现对退化图像添加足够高斯噪声后其分布与干净图像加噪分布对齐，可直接利用预训练扩散模型去噪完成恢复，无需假设退化类型。
2. **生成空间约束机制**：通过fine-tune扩散模型与anchor图像（生成专辑或个人专辑）匹配，约束生成空间以在保持高质量的同时忠实于输入内容。
3. **Skip Guidance生成专辑方法**：提出周期性应用L1 guidance的skip guidance策略，从退化图像生成高质量的生成专辑作为anchor，避免了每步都强指导的伪影问题。
4. **扩展至个性化恢复**：将个人相册直接用作anchor fine-tune模型，实现对特定主体的个性化恢复，显著提升身份保持能力（ArcFace cosine similarity达0.731 vs CodeFormer 0.633）。

## 方法详解
1. **图像投影（Noise Injection）**：对退化观测 $y_0$ 添加高斯噪声得到 $y_K = \sqrt{\alpha_K} y_0 + \sqrt{1-\alpha_K}\epsilon$，当 $K$ 足够大时 $y_K \approx x_K$（干净图像加同等噪声），使得 $p(x_0|y_K) \approx p(x_0|x_K)$，从而可用预训练扩散模型直接去噪。
2. **生成空间约束（Finetuning）**：fine-tune预训练扩散模型以匹配anchor图像分布，使用学习率 $1e^{-5}$、batch size 4，训练3000次迭代（个人相册训练5000次）。
3. **Skip Guidance生成专辑**：从 $y_K$ 出发，周期性（每隔n步）应用梯度更新 $x'_t = x_t - \lambda \nabla_{x_t} ||y_0 - \hat{x}_{0,t}||_2^2$ 生成近似输入的高质量图像，重复多次形成生成专辑（每张输入生成16张图像）。
4. **恢复流程**：将噪声添加至 $y_K$（论文取 $K=200$），然后用fine-tuned模型迭代去噪K步完成恢复，无需额外引导。

## 实验与结果
- **数据集**：Wider-Test (970张)、LFW-Test (1771张)、WebPhoto-Test (407张)、自建Deblur-Test (67张运动模糊图像)
- **评估指标**：FID↓、MUSIQ(Koniq)↑、ArcFace identity score
- **主要结果**：
  - Wider-Test：FID 46.38（最优）、MUSIQ 58.73
  - LFW-Test：FID 56.32（最优）、MUSIQ 60.68
  - WebPhoto-Test：FID 96.44（最优）、MUSIQ 57.71
  - Deblur-Test：FID 135.33（最优）、MUSIQ 60.20，显著超越不建模运动模糊的基线（如GFPGAN 178.40）
- **个性化恢复**：Subject A的identity score 0.731 vs ASFFNet 0.694、CodeFormer 0.633
- **基线对比**：GFPGAN、CodeFormer、VQFR、DR2、ASFFNet
- **结论**：在全部数据集上FID和MUSIQ均取得最优，尤其对未知退化（运动模糊）泛化能力强

## 相关工作脉络
1. **判别式恢复方法**（GFPGAN、CodeFormer、VQFR、RestoreFormer）：依赖合成配对数据训练，退化类型受限于训练集，而本文无需配对数据且对退化类型无假设。
2. **基于模型的优化方法**（RED、DGP、DDRM）：需要已知退化过程H在推理时使用，局限于合成评估；本文完全无需退化模型知识。
3. **扩散模型恢复方法**（IDM、DR2）：IDM需条件注入，DR2结合VQFR；本文仅用无条件扩散模型即可实现。
4. **个性化扩散模型**（DreamBooth、InstantBooth、FastComposer）：侧重文本到图像生成的定制化；本文将其应用于图像恢复的生成空间约束。
5. **参考图像恢复**（ASFFNet）：使用多参考图融合，但生成质量受限；本文通过生成专辑fine-tune实现高质量个性化恢复。

## 局限性与未来方向
1. **单图像恢复需逐图fine-tune**：相比feed-forward方法速度较慢，未来可探索无需fine-tune的生成空间约束方法。
2. **验证集中于类别特定任务**：主要在人像、猫狗等类别上验证，因缺乏高质量自然图像预训练扩散模型；未来可扩展至更广泛生成空间。
3. **K值选择需要权衡**：过小导致分布偏移质量差，过大则输入信息丢失，需经验调参。

## 研究启发与可借鉴点
1. **"噪声对齐"思想可迁移**：添加足够噪声使退化图像与干净图像分布对齐的思路可用于其他逆问题（如去雨、去雾、超分）。
2. **Skip Guidance周期性引导策略**：避免每步强引导的伪影问题，这一技巧可用于条件扩散生成任务。
3. **生成专辑作为约束anchor**：通过skip guidance生成贴近输入的多样化高质量样本作为fine-tuneanchor，替代真实参考图像，为无参考场景提供新思路。
4. **身份保持与质量平衡的验证设计**：使用ArcFace cosine similarity量化身份保持，为个性化恢复提供可复用的评估范式。

## 关键术语表
- **Restoration by Generation**：利用扩散模型采样过程直接生成恢复结果而非学习逆映射的范式
- **Generative Album**：通过skip guidance从退化图像生成的近似高质量图像集合，用作约束生成空间的anchor
- **Personal Album**：同一主体的多张干净参考图像，用于个性化fine-tune
- **Skip Guidance**：周期性而非每步应用L1 guidance的策略，平衡生成质量与输入保真度
- **Blind Face Restoration**：退化类型未知的单张人脸图像恢复任务
- **Identity Score**：基于ArcFace特征余弦相似度量化恢复结果保持输入身份的指标

## 可复现要素
- **数据集**：Wider-Test、LFW-Test、WebPhoto-Test来自公开基准；Deblur-Test从[25]选取67张；FFHQ用于预训练扩散模型；AFHQ Dog/Cat用于动物恢复
- **代码开源**：项目页面 https://gen2res.github.io 提供代码和权重
- **关键超参**：$K=200$、学习率 $1e^{-5}$、batch size 4、迭代3000次（生成专辑）、5000次（个人相册）、生成16张图像组成专辑
- **预训练模型**：FFHQ上预训练的无条件扩散模型
