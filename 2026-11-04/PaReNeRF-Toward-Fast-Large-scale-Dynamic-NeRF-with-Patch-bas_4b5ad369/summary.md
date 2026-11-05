---
title: "PaReNeRF-Toward-Fast-Large-scale-Dynamic-NeRF-with-Patch-bas"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Tang_PaReNeRF_Toward_Fast_Large-scale_Dynamic_NeRF_with_Patch-based_Reference_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 00:44:26"
field: "神经渲染与三维重建"
keywords: ["NeRF", "动态场景重建", "大规模神经渲染", "参考解码器", "Patch采样", "自动驾驶仿真"]
innovations: ["提出基于光流与SSIM的先验信息搜索方法，加速动态场景中参考patch检索", "设计轻量级参考解码器，将高分辨率参考信息融合到低分辨率体渲染特征图上提升重建质量", "采用patch-based采样替代随机射线采样，减少数据I/O开销加速训练"]
benchmarks: ["KITTI", "VKITTI2"]
---

# 论文速读：PaReNeRF: Toward Fast Large-scale Dynamic NeRF with Patch-based Reference

## 一句话总结
本文提出PaReNeRF，一种基于参考解码器的大规模动态神经辐射场加速方法，通过**patch-based采样**和**参考先验信息融合**，在KITTI和VKITTI2数据集上实现比SOTA方法SUDS训练时间减少60%、推理时间减少87%、PSNR提升13%的性能。

## 研究问题与动机
- **训练/推理速度慢**：现有大规模动态场景NeRF方法（如SUDS、Mars）需要高分辨率渲染和跨视频随机射线采样，训练一个9秒场景需超过2天。
- **低分辨率渲染损失质量**：Unisim等方法采用低分辨率特征图+CNN上采样以加速，但导致重建精度下降。
- **动态场景数据稀疏**：自动驾驶采集数据具有单轨迹、视角稀疏的特点，难以直接应用静态小场景的NeRF方法。
- **参考信息利用不足**：现有基于参考信息的NeRF（如pixelNeRF、NeRF-SR）仅适用于静态小场景，未考虑动态场景中的时空调节与搜索效率问题。

## 核心贡献（创新点）
- **Patch-based采样替代随机射线采样**：通过按patch读取数据减少数据加载次数，显著加速训练阶段；与已有方法（如随机采样）的本质区别在于利用了空间局部性减少IO开销。
- **基于光流与结构相似性的先验信息搜索方法**：利用2D光流在前后帧中缩小搜索范围，再滑动patch计算结构相似度（SSIM）寻找最佳参考；与NeRF-SR基于深度图的静态搜索本质不同，适用于动态场景。
- **轻量级参考解码器（Reference Decoder）**：在低分辨率体渲染特征图上融合高分辨率参考patch，基于CARN设计upsampling模块；与Unisim单纯CNN上采样的本质区别是引入了已知视角的详细信息。
- **系统性速度-质量协同优化**：同时实现训练时间↓60%、推理时间↓87%、PSNR↑13%，在KITTI上PSNR达32.64 dB，全面超越SUDS基线。

## 方法详解
- **基础框架**：以SUDS为基线，SUDS将场景分解为static、dynamic、far-field三个分支，使用hash table编码静态/动态辐射场。
- **Patch-based Ray Sampling**：将随机射线采样改为patch采样，设patch大小为$h \times w$，每batch需加载数据$N/(h \times w)$次而非$N$次，大幅减少数据读取时间。
- **低分辨率渲染+上采样**：渲染$414 \times 125$的低分辨率特征图（原始图像$1242 \times 375$），通过3x PixelShuffle上采样。
- **先验信息搜索（训练阶段）**：
  - 对采样patch $P \in \mathbb{R}^{h \times w \times 3}$，利用2D光流$f l_{ij}$计算前后帧中相似区域$Ref_f$和$Ref_b$。
  - 在相似区域内滑动patch，计算SSIM，取最大SSIM的patch作为参考patch RP。
- **先验信息搜索（推理阶段）**：计算体渲染特征图$F$与所有已知视图RGB图像的SSIM，选取最高者作为参考图像$RI$。
- **参考解码器结构**：
  - 特征编码器提取参考图像$RI$的隐藏特征。
  - CARN（Cascading Residual Network）特征编码器提取低分辨率特征图$F$的特征，通过cascading residual模块融合多层信息。
  - 上采样模块：$3 \times 1$卷积层+PixelShuffle实现3x上采样。
  - 特征解码器输出最终RGB图像$I$。
- **优化目标**：沿用SUDS的损失函数组合（重建损失、warping损失、flow损失等），仅将L2光度损失$\mathcal{L}_c(r) = \|C(r) - \hat{C}(r)\|^2$中的$\hat{C}(r)$替换为参考解码器输出的图像，并以patch-wise方式计算损失。

## 实验与结果
- **数据集**：KITTI [12]、Virtual KITTI-2 (VKITTI2) [2]，训练序列最多90个时间步/9秒，图像尺寸$1242 \times 375$，双视角。
- **评估任务**：图像重建（Image Reconstruction）和新视角合成（Novel View Synthesis，NVS），NVS采用75%/50%/25%训练集划分。
- **评估指标**：PSNR、SSIM、LPIPS，以及训练时间（h）和推理时间（s）。
- **主要结果（KITTI图像重建）**：
  | 方法 | PSNR↑ | SSIM↑ | 训练时间(h)↓ | 推理时间(s)↓ |
  |------|-------|-------|-------------|-------------|
  | SUDS | 28.31 | 0.870 | 64 | 12.86 |
  | Ours | **32.64** | **0.933** | **26** | **1.61** |
  - 相比SUDS：PSNR提升13%，训练时间减少60%，推理时间减少87%。
- **NVS结果（KITTI-75%）**：PSNR 25.19 vs SUDS 22.77，SSIM 0.879 vs 0.797，LPIPS 0.075 vs 0.171，全面领先。
- **Ablation（Table 4）**：
  - 仅patch sampling：训练提速但精度下降（PSNR 27.04）。
  - Patch + Encoder-Decoder：推理提速至1.75s，PSNR 29.31。
  - 完整方法（+Reference Decoder）：PSNR达33.23（250K iter）/32.64（125K iter），综合最优。

## 相关工作脉络
- **SUDS [48]**：当前大规模动态场景重建SOTA，将场景分解为static/dynamic/far-field三分支，使用hash table加速，但训练慢（9秒需>2天）、推理慢；本文以此为基础进行加速改进。
- **UniSim [60]**：采用低分辨率特征图+CNN上采样加速，但encoder-decoder结构降低重建精度；本文在此基础上引入参考解码器弥补精度损失。
- **NSG [32] / PNF [22]**：将场景分解为背景和多个移动物体，但需3D bounding box和panoptic标签监督，且显存随物体数量线性增长；本文无需此类人工标注。
- **Mars [55]**：实例感知的大规模动态场景重建方法，训练同样耗时超过2天；本文在速度与质量上均实现突破。
- **NeRF-SR [49]**：利用单一HR参考图通过patch匹配进行超分，但仅适用于静态场景且无法加速渲染；本文的光流+SSIM搜索适配动态场景。
- **pixelNeRF [61] / MVSNeRF [4]**：将参考视图特征投影到查询点，但仅适用于小静态/目标中心场景；本文针对自动驾驶稀疏轨迹的动态场景设计。

## 局限性与未来方向
- **推理时参考图像搜索开销**：虽然训练阶段通过光流缩小了搜索范围，但推理阶段仍需遍历所有已知视图计算SSIM，全文未提及进一步优化策略。
- **真实实时性尚未达成**：论文自述"there are still many open challenges before building truly real-time training and rendering"，当前1.61s/inference尚未达到实时标准。
- **动态物体极端稀疏视角下的表现**：当训练视角比例降至25%时，PSNR从25.19降至21.05（KITTI），稀疏视角下的泛化能力仍有提升空间。
- **依赖光流质量**：先验搜索依赖2D光流估计，在遮挡区域或大位移场景下光流可能不准确，进而影响参考patch选择质量。

## 研究启发与可借鉴点
- **Patch-based采样加速数据I/O**：将随机射线采样改为patch采样，减少对原始图像的重复读取，该方法可迁移至其他NeRF变体的训练加速设计中。
- **光流引导的参考搜索策略**：利用时序光流在视频序列中快速定位相似区域，避免了全图暴力搜索，该思路可应用于其他基于参考的视频生成/重建任务。
- **Reference Decoder设计**：将高分辨率参考信息通过CARN类网络融合到上采样流程中，兼顾速度和质量，其轻量级设计（~毫秒级延迟）值得在SR、超分NeRF等方向复用。
- **实验设计亮点**：同时报告训练时间、推理时间、PSNR/SSIM/LPIPS多维度指标，并做完整的ablation验证各组件贡献，可作为团队实验报告的参考范式。

## 关键术语表
- **NeRF (Neural Radiance Field)**：通过MLP隐式编码连续场景辐射场的神经渲染方法，可实现照片级新视角合成。
- **SUDS (Scalable Urban Dynamic Scenes)**：当前大规模动态场景重建SOTA方法，将场景分解为static/dynamic/far-field三分支，使用hash table编码。
- **Patch-based Sampling**：将随机射线采样改为按空间patch读取数据，减少数据加载次数，加速训练阶段。
- **Reference Decoder**：轻量级上采样网络，利用已知视图的高分辨率参考信息增强低分辨率体渲染特征图的重建质量。
- **Structural Similarity (SSIM)**：衡量两图像结构相似度的指标，本文用于检索最相似的参考patch/图像。
- **CARN (Cascading Residual Network)**：一种高效的超分辨率网络架构，通过cascading residual模块融合多层特征，本文用作参考解码器 backbone。
- **Volume Rendering**：沿射线采样并累积密度和颜色值的渲染过程，本文输出低分辨率特征图而非直接RGB。
- **Novel View Synthesis (NVS)**：从部分训练视角合成未见视角图像的任务，本文在75%/50%/25%训练集比例下均取得最佳结果。

## 可复现要素
- **数据集**：KITTI [12]、VKITTI2 [2]（公开可用）
- **代码开源情况**：论文未明确声明代码开源状态（论文未提及）
- **权重开源情况**：论文未提及
- **关键超参**：
  - 训练迭代数：125,000（主实验）/ 250,000（ablation对比）
  - Batch size：4,096 rays，每batch含16个patch，patch大小16×16
  - 渲染分辨率：414×125（低分辨率特征图）
  - 上采样倍数：3x
  - 优化器：Adam，初始学习率$5 \times 10^{-3}$，衰减至$5 \times 10^{-4}$
  - 输入图像尺寸：1242×375
