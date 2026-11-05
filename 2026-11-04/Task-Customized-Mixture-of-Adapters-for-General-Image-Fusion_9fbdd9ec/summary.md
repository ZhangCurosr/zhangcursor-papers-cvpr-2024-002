---
title: "Task-Customized-Mixture-of-Adapters-for-General-Image-Fusion"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Zhu_Task-Customized_Mixture_of_Adapters_for_General_Image_Fusion_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 08:35:33"
field: "多模态图像融合"
keywords: ["图像融合", "Mixture of Experts", "参数高效微调", "视觉Transformer", "通用融合", "适配器"]
innovations: ["首次将MoE适配器机制引入预训练ViT实现通用图像融合", "提出互信息正则化约束双源特征互补性", "通过TopK稀疏路由实现任务定制化的动态融合强度控制"]
benchmarks: ["LLVIP", "MEFB", "RealMFF/MFI-WHU"]
---

# 论文速读：Task-Customized-Mixture-of-Adapters-for-General-Image-Fusion

## 一句话总结
论文提出 **TC-MoA**（Task-Customized Mixture of Adapters），将 MoE 思想引入预训练 ViT 基础模型，通过任务特定的路由器网络动态选择适配器生成融合 prompt，以极小可学习参数量（仅增加 2.8%）统一处理多模态、多曝光、多焦点三类图像融合任务。

## 研究问题与动机
- **跨任务融合机制差异大**：MEF 关注均衡光照融合，VIF 需保留红外强度+可见纹理，MFF 呈极端不平衡的"极化选择"特性，现有方法难以统一。
- **任务专用方法泛化差**：如 IFCNN 等方法设计复杂任务偏置网络/损失，对其他任务性能下降。
- **现有通用方法顾此失彼**：DeFusion/U2Fusion 等以牺牲任务个体性换取跨任务共性；IFCNN 存在主导任务偏差，未能同时兼容不同任务的融合偏好。
- **缺乏可控性与可扩展性**：现有通用方法难以根据场景动态调整融合强度偏向。

## 核心贡献（创新点）
1. **首次将 MoE 思想引入图像融合**：以适配器作为 expert 对预训练 ViT 进行高效提示融合，仅增加 2.8% 可学习参数即可覆盖多种融合任务。
2. **提出互信息正则化（MIR）**：约束 prompt 满足 $\min|prompt_x + prompt_y - 1|$，确保不同源图像的特征互补性，区别于以往通用的损失设计。
3. **任务定制化路由器网络实现动态融合偏好控制**：每个任务有专属路由器（$G^V, G^E, G^F$），通过 TopK 选择激活适配器，显式建模不同任务的主导强度偏向。
4. **实现 prompt 线性可控性**：通过缩放因子 $\alpha$ 和偏移因子 $\beta$ 可线性调制融合图像的亮度偏向与区域选择性。

## 方法详解
- **骨干网络**：冻结的 MAE-pretrained ViT（Base/Large），分为 encoder 和 decoder 两部分，各 transformer block 间插入 TC-MoA 模块（$\tau=4$）。
- **TC-MoA 结构**：由任务特定路由器库 $\{G^V, G^E, G^F\}$、共享适配器库 $\{A_1,...,A_N\}$（$N=4$）和融合层 $\mathcal{F}$ 组成。
- **Prompt 生成**：
  $$\Phi = L(Cat(f_x, f_y))$$
  $$G(x) = Softmax(TopK(x \cdot W_g + N(0,1) \cdot Softplus(x \cdot W_{noise})))$$
  $$prompt = GAP(Sigmoid(\sum_{i=1}^{N} G(\Phi)_i \cdot A_i(\Phi)))$$
  采用 TopK（$K=2$）稀疏路由，加入高斯噪声增强路由多样性。
- **Prompt 驱动融合**：
  $$h_x = prompt_x \cdot f_x + S_x, \quad h_y = prompt_y \cdot f_y + S_y$$
  $$f_{TC-MoA} = \mathcal{F}(h_x + h_y)$$
  其中 $S_x, S_y$ 为输入无关的可学习源嵌入，$\mathcal{F}$ 为卷积层消除棋盘伪影。
- **互信息正则化（MIR）**：
  $$\min |prompt_x + prompt_y - 1|$$
  强制两源 prompt 之和趋近 1，保证互补性。
- **任务定制损失**：
  - VIF：$\mathcal{L}_{MaxPixel} + \mathcal{L}_{MaxGrad} + \mathcal{L}_{ssim}$
  - MEF：$\mathcal{L}_{AvgPixel} + \mathcal{L}_{MaxGrad} + \mathcal{L}_{mefsse}$
  - MFF：$\mathcal{L}_{MaskPixel} + \mathcal{L}_{MaskGrad}$（每 patch 仅选梯度最大源计算损失，防止散焦边缘残留）
  - 辅助损失 $\mathcal{L}_{aux}$ 避免适配器学习不平衡。

## 实验与结果
- **数据集**：VIF→LLVIP（训练 12025 对，测试 70 对）；MEF→SCIE 训练，MEFB 测试；MFF→RealMFF + MFI-WHU 训练，MFIF 测试。
- **基线**：任务专用方法（Densefuse、CDDFuse、IFCNN 等）+ 通用方法（U2Fusion、DeFusion、PMGI 等）。
- **VIF（LLVIP）**：TC-MoA 在所有通用方法中全面领先；$Q_{cv}=423.773$（更低越好）、$EN=7.428$、MS-SSIM=0.949，超越 CDDFuse（当前最强 VIF 方法）的 $Q_{cv}=495.473$ 和 MS-SSIM=0.933。
- **MEF（MEFB）**：PSNR=57.213、MEF-SSIM=0.964，超越 IFCNN（PSNR=57.195、MEF-SSIM=0.943）。
- **MFF**：NMI=0.875、$Q_{cv}=36.512$、$Q_w=0.891$，显著优于 IFCNN（NMI=0.847、$Q_{cv}=44.373$）。
- **效率**：Base 版本可训练参数仅 3.87M（Large 为 9.58M），VIF 640×512 输入 FPS=3.33，较冻结 ViT 加速 178%。

## 相关工作脉络
1. **U2Fusion（Ma et al., 2020）**：自监督通用融合框架，学习跨任务共性但忽略任务个体性；本文通过 MoE 适配器机制在保留共性的同时捕捉任务个性化。
2. **DeFusion（Liang et al., ECCV 2022）**：同样自监督学习融合共性，本文通过互信息正则化和任务定制路由弥补其缺乏任务特异性的不足。
3. **CDDFuse（Zhao et al., CVPR 2023）**：当前最强 VIF 专用方法，采用双分支相关驱动分解；本文证明单一通用模型可在 VIF 上超越专用方法。
4. **IFCNN（Zhang et al., 2020）**：首个通用融合框架，但存在主导任务偏差；本文通过路由器差异化解决此问题。
5. **AdaptFormer（Chen et al., NeurIPS 2022）**：视觉 Transformer 的 PEFT 适配器方法，本文将其思想扩展至图像融合领域并证明有效性更优。
6. **SwinFusion（Ma et al., 2022）**：基于 Swin Transformer 的统一框架，本文与 SwinFusion 的定位差异在于采用 MoE 适配器替代固定融合策略，增强动态适应性。

## 局限性与未来方向
- **骨干网络参数量仍较大**：冻结 ViT-Large 总参数 339M，虽可训练部分仅 9.58M，但推理内存开销仍高于轻量级专用方法。
- **仅验证了三类融合任务**：泛化到医学图像融合、全色 Sharpening 等需通过 prompt 调参，缺乏端到端自适应迁移。
- **超参 N（适配器数）存在最优值**：N 过大性能反而下降，缺乏理论指导。
- **无监督损失依赖人工设计**：不同任务需定制化损失，未探索完全免设计的融合方案。

## 研究启发与可借鉴点
1. **MoE+PEFT 范式迁移**：将路由器+适配器结构用于其他视觉融合/生成任务（如去噪+超分的联合模型），具备可扩展性。
2. **互信息正则化的设计思路**：约束输出互补性而非直接监督融合结果，可推广至其他多源信息融合场景。
3. **Prompt 线性可控性机制**：通过 $\alpha, \beta$ 调制融合强度的方法，为下游应用（如医疗图像按需强调某模态）提供可解释接口。
4. **TopK 稀疏路由 + 噪声注入**：增强路由器多样性和训练稳定性，可复用于多任务视觉模型的共享-个性化参数分配。
5. **源嵌入（Source Embedding）设计**：引入与输入无关的可学习偏置 $S_x, S_y$ 区分不同源，简单有效，可用于多传感器特征对齐。

## 关键术语表
- **TC-MoA**：Task-Customized Mixture of Adapters，本文提出的任务定制化适配器混合架构。
- **MIR（Mutual Information Regularization）**：互信息正则化，约束双源 prompt 之和趋近 1 以保证特征互补。
- **Prompt-driven Fusion**：基于 prompt 驱动的融合，将生成的 prompt 作为源特征的缩放因子实现自适应融合。
- **TopK Routing**：稀疏路由机制，仅激活 TopK 个适配器（本文 K=2），降低计算量并增强任务特异性。
- **Source Embedding**：与输入无关的可学习偏置参数 $S_x, S_y$，为不同源图像提供来源相关性偏移。
- **Dominant Intensity Bias**：主导强度偏向，描述不同融合任务中各源图像贡献度的偏好差异。
- **PEFT（Parameter-Efficient Fine-Tuning）**：参数高效微调，仅训练少量参数即可适配下游任务。
- **LLVIP**：低光可见-红外配对数据集，用于 VIF 任务评测。

## 可复现要素
- **数据集**：LLVIP、MEFB、SCIE、RealMFF、MFI-WHU、MFIF（部分数据集需申请授权）
- **代码开源**：https://github.com/YangSun22/TC-MoA
- **权重开源**：论文未明确声明，代码仓库中可能有提供
- **关键超参**：$N=4$（适配器数）、$K=2$（激活数）、$\tau=4$（插入间隔）、batch size=3、lr=$1.5\times10^{-4}$、20 epochs、输入尺寸 448×448、EMA 优化路由器与适配器
- **骨干网络**：MAE-pretrained ViT-Base / ViT-Large
