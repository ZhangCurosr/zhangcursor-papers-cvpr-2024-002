---
title: "PeLK-Parameter-efficient-Large-Kernel-ConvNets-with-Peripher"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_PeLK_Parameter-efficient_Large_Kernel_ConvNets_with_Peripheral_Convolution_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 15:20:17"
---

# 论文速读：PeLK-Parameter-efficient-Large-Kernel-ConvNets-with-Peripher

## 一句话总结
本文受人类周边视觉启发，提出“周边卷积（Peripheral Convolution）”将大核稠密卷积的参数复杂度从 $O(K^2)$ 降至 $O(\log K)$，首次将 CNN 卷积核尺寸扩展至惊人的 $101 \times 101$，并在 ImageNet 分类、ADE20K 分割与 MS COCO 检测任务上全面超越 Swin、ConvNeXt、RepLKNet 与 SLaK 等主流架构。

## 研究问题与动机
1. **大核卷积的平方复杂度瓶颈**：纯 CNN 配合大感受野表现优异，但稠密网格卷积参数量随核尺寸呈 $O(K^2)$ 增长，直接放大核尺寸会导致参数量激增与优化困难。
2. **现有妥协方案的性能上限**：RepLKNet 依赖并行小核缓解优化问题，SLaK 采用条纹卷积（$K \times N + N \times K$）将复杂度线性化并扩至 $51 \times 51$，但条纹形式缺乏稠密卷积的全局空间感知能力，且性能已开始饱和。
3. **核心科学问题**：是否存在一种设计，既能保留稠密网格卷积的形式优势，又能大幅削减参数，从而持续放大核尺寸以获得更多性能增益？
4. **生物视觉的启示**：人类视觉系统中中央凹区域分辨率极高，而周边区域感光细胞稀少、感知模糊但能高效捕捉宏观上下文，为参数空间冗余共享提供了天然原型。

## 核心贡献（创新点）
1. **提出周边卷积（Peripheral Convolution）范式**：通过空间维度的指数化参数共享将大核参数量削减超 90%，复杂度降至 $O(\log K)$，与已有的稀疏化或条纹卷积策略形成本质差异。
2. **设计“聚焦-模糊（Focus-and-Blur）”共享网格**：中心区域保持细粒度独立参数以保留细节感知，外围区域采用大范围共享以利用空间冗余，区别于均匀共享或固定步长共享。
3. **引入核内位置嵌入（Kernel-wise Positional Embedding）**：以近乎零额外参数（仅 0.05% FLOPs）的代价引入相对位置偏置，优雅弥补大范围共享导致的局部细节模糊问题。
4. **构建 PeLK 架构并验证极限扩展能力**：首次成功将 CNN 大核扩展至 $101 \times 101$ 稠密卷积，在分类、分割、检测三大任务上取得 SOTA，证明生物启发机制可显著推动现代纯 CNN 设计。

## 方法详解
1. **参数共享网格（Sharing Grid）**：用小核 $\mathbf{w}_\theta \in \mathbb{R}^{c_{in} \times c_{out} \times k' \times k'}$ 参数化大核 $\mathbf{w} \in \mathbb{R}^{c_{in} \times c_{out} \times k \times k}$，共享网格 $S = [s_0, ..., s_{k'-1}]$ 满足 $\sum s_i = k$，将 $k \times k$ 区域划分为 $k' \times k'$ 个子块，块内所有位置共享同一权重。
2. **聚焦-模糊机制**：将共享网格重构为关于中心对称的形式 $[\bar{s}_{-r}, ..., \bar{s}_0, ..., \bar{s}_r]$。中心半径 $r_c$ 内设置 $\bar{s}_i = 1$（不共享），外围设置 $\bar{s}_i = m^{(|i|-r_c)}$（大范围共享），$m$ 默认取 2。
3. **核内位置嵌入**：卷积输出修正为 $Y(x,y) = \sum_{i,j} \mathbf{w}(i,j) \cdot (X(x+i, y+j) + \mathbf{h}(i,j))$，其中 $\mathbf{h}$ 为截断正态初始化的可学习位置偏置矩阵，同一 Stage 内所有大核共享 $\mathbf{h}$。
4. **部分周边卷积（Partial Peripheral Convolution）**：借鉴 Inception 风格，将输入特征沿通道拆分为 $X_{conv}$（$\frac{3}{8}C_{in}$ 通道）与 $X_{id}$，前者经周边卷积处理，后者走恒等映射，最终
