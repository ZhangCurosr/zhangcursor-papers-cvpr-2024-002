---
title: "ZeroNVS-Zero-Shot-360-Degree-View-Synthesis-from-a-Single-Im"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Sargent_ZeroNVS_Zero-Shot_360-Degree_View_Synthesis_from_a_Single_Image_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 16:38:24"
field: "单图像3D生成与新视角合成"
keywords: ["Novel View Synthesis", "Diffusion Model", "3D Generation", "Score Distillation Sampling", "Single-Image 3D"]
innovations: ["6DoF+1相机条件参数化与viewer-centric尺度归一化", "SDS锚定机制提升背景多样性", "零样本360度场景新视角合成"]
benchmarks: ["DTU", "Mip-NeRF 360", "CO3D", "RealEstate10K", "ACID"]
---

# 论文速读：ZeroNVS: Zero-Shot 360-Degree View Synthesis from a Single Image

## 一句话总结
本文提出 ZeroNVS，一个3D感知扩散模型，能够从单张真实图像零样本生成360度一致的新视角图像，通过新的相机条件参数化、深度归一化方案以及SDS锚定机制，解决了真实场景背景多样性和尺度模糊问题。

## 研究问题与动机
1. **现有方法局限于单一物体场景**：已有单图像新视角合成方法（如Zero-1-to-3）主要针对Objaverse-XL等单一物体数据集设计，忽略背景，无法处理真实世界的多对象复杂场景。
2. **相机表示不足**：传统3DoF（仰角、方位角、半径）参数化无法表达真实场景中6DoF任意相机位姿，且对roll角和空间任意位置无能为力。
3. **尺度模糊问题**：单目输入存在固有尺度歧义，不同数据集（COLMAP vs ORB-SLAM）产生的深度尺度不一致，导致模型难以泛化。
4. **SDS蒸馏导致背景多样性丧失**：Score Distillation Sampling (SDS) 在生成长距离（如180度）新视角时倾向于产生单调灰色背景，缺乏内容多样性。

## 核心贡献（创新点）
1. **新型6DoF+1相机条件参数化**：提出相对位姿加视场角的6DoF+1表示，取代Zero-1-to-3的3DoF表示，使模型能够处理真实场景中的任意相机位姿。
2. **基于观察者中心尺度的归一化方案**：设计viewer-centric深度归一化（取输入视图第20分位数），避免多视图聚合导致的尺度歧义和训练数据分布不一致问题。
3. **SDS锚定机制**：先用DDIM采样多样化锚定视图，再用SDS以最近视图为条件进行蒸馏，显著提升背景多样性（用户研究中创意性评分82%偏好）。
4. **建立新评估基准**：将Mip-NeRF 360数据集适配为单图像新视角合成基准，证明ZeroNVS在复杂360度场景下的零样本泛化能力。

## 方法详解
**整体框架**：两阶段方法——(1) 在混合场景数据集上训练2D条件扩散模型；(2) 通过SDS蒸馏将扩散先验提升至3D NeRF表示。

**相机条件参数化演进**：
- **M_Zero-1-to-3**：`P(E_i) - P(E_j)`，仅用3DoF（仰角、方位角、半径）
- **M_6DoF+1**：`[E_i^{-1}E_j, f]`，相对位姿+视场角，刚性变换不变
- **M_6DoF+1,norm**：相机位置按平均范数归一化
- **M_6DoF+1,agg**：基于所有深度图第10分位数聚合的尺度归一化（Stereo Magnification思路）
- **M_6DoF+1,viewer**（最终方案）：`[R(E_i^{-1}E_j, 1/q_i), f]`，仅用输入视图深度第20分位数$q_i = Q_{20}(\bar{D}_i)$

**关键设计要点**：
- 使用离线深度估计器填充深度孔洞，但推理时不使用
- SDS锚定流程：先用DDIM均匀采样k个锚定视图→SDS优化时以最近视图（输入或采样视图）为条件
- 初始化自预训练Zero-1-to-3-XL，替换条件模块

## 实验与结果
**训练数据**：CO3D、RealEstate10K、ACID混合数据集（各1/3均匀采样），256×256分辨率。

**DTU零样本评测**（Table 1）：
- ZeroNVS LPIPS: **0.380**，超越Fine-tuned方法（NeRDi 0.421）
- PSNR 13.55，SSIM 0.469

**Mip-NeRF 360零样本评测**（Table 2）：
- ZeroNVS LPIPS: **0.625**，优于Zero-1-to-3（0.667）和PixelNeRF（0.718）

**消融实验**（Table 4）：
- M_6DoF+1,viewer在各数据集上均达到最优或接近最优
- 各训练数据集贡献（Table 3）：移除任一数据集均导致性能下降

**SDS锚定效果**（用户研究，21人）：
- 真实性偏好：78%，创意性：82%，总体偏好：80%

## 相关工作脉络
1. **Zero-1-to-3**：单物体3DoF相机参数化，本文扩展至6DoF场景级表示，并解决尺度模糊。
2. **DreamFusion/SDS**：文本到3D的Score Distillation Sampling，本文发现其在场景背景生成中的模式坍塌问题并提出锚定机制。
3. **PixelNeRF/DietNeRF**：基于图像的NeRF重建方法，依赖单一几何先验，缺乏生成多样性。
4. **GeNVS**：仅支持特定物体类别的360度合成，本文面向通用场景。
5. **RealFusion**：单图像360度重建，但聚焦物体而非复杂背景场景。
6. **NeRDi**：语言引导的单视角NeRF合成，本文纯视觉条件，零样本泛化更强。

## 局限性与未来方向
1. **计算成本**：SDS蒸馏耗时较长，难以应用FID等生成指标评估多样性。
2. **深度估计依赖**：训练时使用离线深度估计器填充孔洞，虽然推理不依赖，但深度质量影响尺度归一化效果。
3. **分辨率限制**：当前在256×256分辨率下评估，高分辨率场景生成能力待验证。
4. **正向相机场景主导**：DTU数据集主要为forward-facing相机 rig，对任意位姿泛化仍需更多验证。

## 研究启发与可借鉴点
1. **相机条件参数化设计**：6DoF+1表示的刚性变换不变性和尺度归一化思路可迁移至其他3D生成任务。
2. **SDS锚定机制**：利用DDIM采样生成多样化条件再指导SDS的策略，可推广至其他扩散模型3D蒸馏场景。
3. **viewer-centric尺度归一化**：仅依赖输入视图深度的方案避免了多视图聚合的歧义，值得在单目深度估计任务中参考。
4. **混合数据集训练策略**：CO3D+RealEstate10K+ACID的异构数据混合方案，为场景级3D生成提供了可扩展的数据配方。

## 关键术语表
**Score Distillation Sampling (SDS)**：利用预训练2D扩散模型对3D渲染进行梯度指导的技术，实现文本/图像到3D的蒸馏。
**NeRF (Neural Radiance Fields)**：基于神经隐式表示的场景渲染方法，通过MLP编码3D点的位置和视角。
**Mip-NeRF 360**：包含360度全景场景的基准数据集，用于评估无界场景的视角合成能力。
**DDIM (Denoising Diffusion Implicit Models)**：加速扩散模型采样的确定性采样方法，相比DDPM更高效。
**LPIPS**： Learned Perceptual Image Patch Similarity，基于深度学习感知的图像相似性度量，越低越好。
**CO3D**：Common Objects in 3D数据集，包含大量真实世界物体的多视角图像和深度。
**RealEstate10K**：室内视频数据集，提供丰富的真实场景图像序列。
**ACID**：Infinite Nature数据集，提供自然场景的无限视角生成数据。

## 可复现要素
- **数据集**：CO3D、RealEstate10K、ACID（公开）、DTU（公开）、Mip-NeRF 360（公开）
- **代码/模型**：论文声明代码和模型可用（链接见原文），基于Threestudio框架
- **关键超参**：分辨率256×256（DTU用400×300）、训练数据来源比例1:1:1、深度分位数Q20
- **预训练权重**：Zero-1-to-3-XL
- **实现细节**：PyTorch、噪声调度 annealing、自定义NeRF网络（Mip-NeRF 360 + Instant-NGP特性）
