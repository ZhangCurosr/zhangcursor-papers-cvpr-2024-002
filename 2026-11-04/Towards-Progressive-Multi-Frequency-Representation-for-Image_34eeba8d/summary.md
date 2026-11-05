---
title: "Towards-Progressive-Multi-Frequency-Representation-for-Image"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Xiao_Towards_Progressive_Multi-Frequency_Representation_for_Image_Warping_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 08:36:26"
field: "图像几何变换与增强"
keywords: ["Image Warping", "Multi-Frequency Representation", "Gabor Wavelet", "Super-Resolution", "Frequency Domain Learning", "Geometric Transformation"]
innovations: ["提出渐进式多频表示学习框架MFR，通过堆叠频域模块实现由粗到细的图像扭曲生成", "引入可学习Gabor小波滤波器，利用其时频局部化特性同步捕捉局部空间与频率特征"]
benchmarks: ["DIV2KW", "Set5W", "Set14W", "B100W", "Urban100W", "Flick360"]
---

# 论文速读：Towards-Progressive-Multi-Frequency-Representation-for-Image-Warping

## 一句话总结
本文提出了MFR（Multi-Frequency Representation），一种面向图像扭曲任务的渐进式多频表示学习方法，通过构建可学习的Gabor小波滤波网络从不同频率子带渐进提取图像特征，显著提升了图像扭曲的高频细节生成质量与分布外泛化能力。

## 研究问题与动机
- 传统插值方法（双三次等）在图像扭曲任务中会产生锯齿和模糊伪影，降低图像质量。
- 现有基于神经网络的扭曲方法（如SRWarp、LTEW）难以有效捕捉变形区域的局部变化，生成图像存在畸变且缺乏高频细节。
- 现有方法在处理分布外（out-of-distribution）数据时泛化性能较差，限制了实际应用场景。
- 如何在扭曲过程中同时保留局部高频细节并提升跨域泛化能力，是当前图像扭曲任务亟待解决的关键问题。

## 核心贡献（创新点）
1. 提出渐进式多频表示学习框架（MFR），通过堆叠的频域学习模块从不同频率子带逐层提取图像特征，实现由粗到细的图像生成过程；与SRWarp/LTEW等单阶段生成方法相比，MFR通过多频带分解实现更精细的局部细节恢复。
2. 引入可学习的Gabor小波滤波器替代传统SIREN或高斯滤波器，利用其在空域和频域上的紧支集特性同时捕捉局部空间-频率信息；与已有频率学习方法相比，Gabor滤波器能更有效地提取变形区域的局部高频变化。
3. 设计了门控机制与短连接结构，将输入的空间表征与学习的频率表征自适应融合；与直接使用原始特征的方法相比，门控机制能够动态控制信息流动，提升特征利用率。
4. 系统验证了MFR在homography变换、ERP到透视投影、非对称图像超分辨率三类任务上的有效性，尤其在分布外场景下展现出显著优于SOTA方法的泛化能力。

## 方法详解
**整体架构**：MFR包含两个阶段——特征编码阶段和频域学习阶段。

**特征编码**：
- 使用预训练的超分辨率模型（RRDB或EDSR）将输入图像投影到隐空间，得到特征图 $X \in \mathbb{R}^{W \times H \times d}$
- 结合局部纹理估计器（LTE）融合坐标信息（相对位置）和几何信息（曲率）

**渐进式频域学习**：
- 将隐特征展平为向量序列 $\{x_n\}_{n=1}^{N}$，其中 $N=H \times W$
- 堆叠L个频域学习模块 $\mathcal{F}_\ell$，每个模块通过滤波层提取频域表示 $z_i^{(\ell)}$
- 递推公式：$z_i^{(0)} = \mathcal{G}_{\theta_0}(x_i) + x_i$，$z_i^{(\ell+1)} = \mathcal{G}_{\theta_\ell}(x_i) \otimes (W^{(\ell)}z_i^{(\ell)} + b^{(\ell)})$
- 当滤波层采用正弦函数时，第L层可等价表示为多个正弦函数的线性组合，总项数 $N_{sine}$ 随模块数量指数增长

**Gabor小波滤波器**：
- 定义：$\mathcal{F}_{Gabor}(x) = e^{-\frac{(x-x_0)^2}{\alpha^2}} e^{-i\omega(x-x_0)}$
- 关键参数 $\alpha$（衰减率）和 $\omega$（调制速率）通过学习获得
- 相比SIREN和高斯滤波器，Gabor在空域和频域均具有紧支集，更适合捕捉局部变形区域的高频细节

**解码与输出**：
- 每个模块的输出：$y_i^{(\ell)} = f_{\phi_\ell}(g_\ell(z_i^{(\ell)}) + x_i)$，其中 $g_\ell$ 为一维门控机制
- 最终图像：$y = y_{bic} + \sum_{\ell=1}^{L} y^{(\ell)}$，其中 $y_{bic}$ 为双三次插值的粗尺度图像
- 采用 $\ell_2$ 损失函数，训练600轮，Adam优化器，初始学习率 $2\times10^{-4}$，余弦退火策略

## 实验与结果
**数据集**：DIV2K（训练集，800张高分辨率图像），评测基准包括DIV2KW、Set5W、Set14W、B100W、Urban100W

**评测设置**：区分in-scale（缩放因子在训练集中）和out-of-scale（不在训练集中）两种设置

**Homography变换结果（mPSNR，dB）**：
| 方法 | DIV2KW(is/os) | Set5W(is/os) | Urban100W(is/os) |
|------|---------------|--------------|-------------------|
| LTEW-RRDB | 31.10/26.92 | 38.20/31.07 | 29.50/24.25 |
| MFR-RRDB(Ours) | 31.18/27.12 | 38.23/31.19 | 29.68/24.51 |
- MFR在out-of-scale设置下对DIV2KW和Urban100W分别提升0.20dB和0.26dB，超越第二名的LTEW

**非对称图像超分辨率结果（PSNR，dB）**：
- 以RCAN为骨干网络，在in-scale和out-of-scale设置下全面优于MetaSR、ArbSR、LTEW
- 在Urban100数据集（大尺度因子）上优势尤为明显

**ERP到透视投影**：
- 使用Flick360验证集，无GT情况下通过视觉对比验证
- MFR生成的图像失真更少，细节更清晰

**消融实验**：
- Gabor滤波器对比：learnable Gabor(30.88/26.90) > static Gabor(30.86/26.86) > SIREN(30.87/26.81)，out-of-scale差异显著
- 网络结构验证：短连接和门控机制均有效，完整结构(DIV2KW: 30.88, Urban100W: 29.68)性能最优

## 相关工作脉络
- **SRWarp [41]**：将图像扭曲视为广义超分辨问题，利用预训练SR模型的自适应扭曲层；不足在于生成大尺度图像时泛化能力差。
- **LTEW [22]**：将图像表示视为连续空间，结合傅里叶特征和坐标MLP生成扭曲图像；MLP表征能力有限，难以捕捉局部变化导致畸变。
- **SIREN [40]**：使用正弦激活函数的隐式神经表示；MFR借鉴其频率学习思想，但通过Gabor滤波器增强局部特征捕捉能力。
- **MetaSR [15] / ArbSR [44]**：面向任意尺度超分辨的网络；MFR专注于扭曲任务，引入频域渐进学习机制。
- **Fourier Features [42]**：用正弦编码提升隐式表示的高频学习能力；MFR采用可学习滤波器而非固定编码。
- **频域学习方法**（如FsDR [16]、FDA [50]）：在域泛化和风格迁移中有应用；MFR将其引入扭曲任务，关注局部高频细节生成。

## 局限性与未来方向
- 方法依赖预训练SR模型作为骨干网络，可能限制其在特定领域的适应性。
- 频域模块堆叠层数L和维度d需要针对具体任务调整，缺乏自动搜索机制。
- 实验主要集中在静态图像扭曲，未探索视频序列或动态场景的应用。
- Gabor滤波器的参数学习可能在高复杂度场景下增加计算开销。
- 未来可探索将其应用于视频扭曲、3D点云配准、医学图像配准等更广泛的几何变换任务。

## 研究启发与可借鉴点
- **渐进式频域分解策略**：通过堆叠滤波器逐步提取不同频率子带，可迁移至其他需要多尺度细节生成的任务（如图像修复、去噪）。
- **可学习Gabor滤波器的设计思路**：将域知识（Gabor的时频局部化特性）与神经网络结合，适用于任何需要局部空间-频率联合建模的任务。
- **门控机制与短连接的组合**：用于自适应融合空间特征与频域特征，可推广至特征融合模块设计。
- **out-of-scale评估设置**：为模型泛化能力评估提供了更有意义的基准，建议在其他扭曲/超分研究中沿用。
- **与团队方向的结合点**：若团队关注频域学习、几何变换或超分辨率，MFR的多频表示框架和Gabor滤波器可作为核心组件集成到现有pipeline中。

## 关键术语表
**Image Warping**：通过几何变换将图像像素映射到新坐标系的计算机视觉基础任务，用于改变图像外观。
**Multi-Frequency Representation**：从输入图像的不同频率子带学习特征的表示方法，用于捕捉多尺度细节。
**Gabor Wavelet Filter**：兼具空域和频域局部化特性的滤波器，能有效提取局部频率和空间信息。
**Coarse-to-Fine Generation**：由粗到细的生成策略，先生成低分辨率图像再逐步细化高频细节。
**Out-of-Scale Setting**：评估模型在训练集未见的缩放因子下的泛化能力。
**mPSNR (masked PSNR)**：针对扭曲任务设计的评测指标，忽略边界mask区域的峰值信噪比。
**SIREN**：使用正弦激活函数的隐式神经表示网络，具有强大的高频函数学习能力。
**Gate Mechanism**：一维注意力机制，自适应控制频域特征与空间特征的信息融合比例。

## 可复现要素
- **数据集**：DIV2K（公开）、Flick360（公开）、Set5/Set14/B100/Urban100（公开）
- **代码**：已开源，地址 https://github.com/junxiao01/MFR
- **权重**：论文未明确说明是否开源预训练权重
- **关键超参**：训练600轮、batch size=16、patch size=48×48、初始学习率 $2\times10^{-4}$、Adam($\beta_1=0.9, \beta_2=0.999$)、余弦退火、$\ell_2$ 损失
- **骨干网络**：RRDB/EDSR（用于特征编码），RCAN（用于超分对比实验）
