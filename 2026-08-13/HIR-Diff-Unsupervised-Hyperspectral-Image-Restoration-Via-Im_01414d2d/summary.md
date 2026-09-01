---
title: "HIR-Diff-Unsupervised-Hyperspectral-Image-Restoration-Via-Im"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Pang_HIR-Diff_Unsupervised_Hyperspectral_Image_Restoration_Via_Improved_Diffusion_Models_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 12:24:29"
---

# 论文速读：HIR-Diff: Unsupervised Hyperspectral Image Restoration Via Improved Diffusion Models

## 一句话总结
本文提出一种无监督高光谱图像（HSI）复原框架 HIR-Diff，将干净HSI分解为低秩系数矩阵与降维图像，利用预训练扩散模型结合TV先验引导与指数噪声调度，在去噪、超分辨率与修复任务上实现了高精度且仅需20步采样的快速恢复。

## 研究问题与动机
- **核心问题**：HSI在采集过程中易受传感器与大气干扰产生噪声、降采样或像素缺失，需从单次退化观测中无监督复原。
- **现有方法不足**：
  1. 传统模型驱动方法依赖手工先验（低秩、TV等），难以充分刻画HSI复杂结构，且优化过程计算昂贵、易陷局部最优。
  2. 监督深度学习方法泛化性差，且HSI高质量配对数据稀缺，难以支撑大规模训练。
  3. 现有无监督扩散方法（如 DDS2M）虽实现数据集自适应，但未利用可用数据集先验，且每个新数据集均需漫长的自监督训练过程。

## 核心贡献（创新点）
1. **提出基于预训练扩散的无监督HSI复原框架**：将HSI建模为降维图像与系数矩阵的张量积，利用预训练扩散模型恢复低谱维主体，并设计含TV先验的引导采样函数。（与 DDS2M 等纯自监督方法相比，显式引入低秩张量分解结构先验，彻底免去逐数据集重新训练。）
2. **基于 SVD 与 RRQR 的鲁棒系数矩阵估计**：利用秩-K SVD 提取频谱结构，并通过秩揭示QR分解优选波段索引，使系数矩阵估计对噪声敏感度高且数值稳定。（与最小二乘直接拟合或均匀间隔选带相比，兼顾了抗噪性与矩阵条件数优化。）
3. **指数噪声调度加速扩散采样**：针对条件引导下噪声非均匀衰减的特性，设计前期陡降、后期平缓的 $\bar{\alpha}_t$ 曲线，使采样过程快速收敛并精细修复细节。（相比线性/余弦调度，在20步采样下实现约5倍加速，PSNR/SSIM几乎无损。）

## 方法详解
- **整体流程**：给定退化 HSI $\mathcal{Y}$，先通过 SVD+RRQR 预估计系数矩阵 $\mathbf{E}\in\mathbb{R}^{B\times K}$；再以 $\mathcal{Y}$ 和 $\mathbf{E}$ 为条件，利用改进 DDIM 逐步采样生成降维图像 $\mathcal{A}\in\mathbb{R}^{H\times W\times K}$；最终干净图像由 $\mathcal{X}=\mathcal{A}\times_3\mathbf{E}$ 重构。
- **系数矩阵估计**：对退化图做秩-$K$ SVD：$\mathbf{Y}_{(3)}^\mathrm{T}=(\mathbf{U}\mathbf{S})\mathbf{V}^\mathrm{T}$。定义选中波段对应的子矩阵 $\mathbf{V}_s$，则 $\mathbf{E}=\mathbf{V}\mathbf{V}_s^{-1}$。RRQR 通过列置换最大化 $|\det(\mathbf{V}_s)|$，避免波段高度相关导致的病态逆运算。
- **扩散引导采样**：反向过程采用梯度引导策略 $\hat{\epsilon}_\theta = \epsilon_\theta + s(t)\nabla_{\mathcal{A}_t}\mathcal{L}(\hat{\mathcal{A}}_0,\mathbf{E},\mathcal{Y})$，其中 $\hat{\mathcal{A}}_0$ 由 DDIM 公式由当前加噪图估计。
- **引导损失函数**：$\mathcal{L} = \lambda\|\mathbf{H}(\hat{\mathcal{A}}_0\times_3\mathbf{E})-\mathcal{Y}\|_F^2 + \beta\|\hat{\mathcal{A}}_0\times_3\mathbf{E}\|_{\mathrm{TV}}$。第一项保证重构结果与观测一致，第二项利用 TV 正则抑制扩散生成中的块状伪影并保留边缘。
- **指数噪声调度**：$\bar{\alpha}_t = \mathrm{normalize}(e^{-kt/T})$。前期 $\bar{\alpha}_t$ 快速上升使随机噪声迅速贴近观测结构，后期增长缓慢让扩散先验专注细节细化，20 步即可稳定收敛。

## 实验与结果
- **数据集**：Washington DC Mall ($256\times256\times191$)、Houston ($256\times256\times124$)、Salinas ($128\times128\times190$)。
- **评估指标**：PSNR、SSIM；对比基线涵盖模型驱动（BM4D、NGMeet、ETPTV等）、监督DL（T3SC、SST、MCNet等）及无监督DL（DDS2M、DIP2d/3d）。
- **去噪结果**：WDC Mall (σ=50) 取得 PSNR 40.77 / SSIM 0.94，超越 DDS2M (39.13/0.93) 约 1.64 dB，耗时仅 17s（DDS2M 需 3132s）。
- **超分结果**：Salinas ×2 尺度 PSNR 39.44 / SSIM 0.91，全面领先；×8 尺度仍达 34.48 / 0.82，较次优方法提升约 1–2 dB。
- **修复结果**：Salinas (掩码率0.9) PSNR 37.61 / SSIM 0.90，优于 DDS2M (36.80/0.89)，耗时 16s vs 1354s。
- **最强结果与提升**：去噪 Salinas (σ=50) PSNR 41.95 为全表最高；超分 Salinas ×2 PSNR 39.44 较监督基线 MCNet 提升约 3.4 dB，且推理速度比自监督扩散基线快两个数量级。

## 相关工作脉络
- **DDRM (NeurIPS 2022)**：对线性逆问题通过退矩阵 SVD 在谱空间做扩散复原。本文定位：DDRM 面向自然图像且仅支持线性退化；本文将其推广至多变体 HSI 的非线性/复合退化场景，并解决谱维不匹配问题。
- **DDS2M (ICCV 2023)**：HSI 自监督扩散模型，仅依赖退化图训练。本文定位：DDS2M 泛化强但需逐数据集重新优化网络；本文利用公开 RGB 遥感扩散先验+低秩分解，实现零训练即时推理。
- **NGMeet (CVPR 2019) / BM4D (TIP 2012)**：经典模型驱动去噪方法，依赖非局部变换与全变分先验。本文定位：传统方法先验主观且表达能力受限；本文以数据驱动扩散先验替代手工正则，在保持无监督属性的同时显著提升纹理与边缘恢复。
- **低秩扩散 pansharpening (arXiv 2023)**：首次将预训练扩散用于多光谱/全色融合的低秩分解恢复。
