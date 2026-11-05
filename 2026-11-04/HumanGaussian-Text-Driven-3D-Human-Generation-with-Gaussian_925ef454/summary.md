---
title: "HumanGaussian-Text-Driven-3D-Human-Generation-with-Gaussian"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_HumanGaussian_Text-Driven_3D_Human_Generation_with_Gaussian_Splatting_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 00:41:27"
---

# 论文速读：HumanGaussian-Text-Driven-3D-Human-Generation-with-Gaussian

## 一句话总结
提出 HumanGaussian，一种基于 3D Gaussian Splatting 的高效文本驱动 3D 人体生成框架，通过 SMPL-X 结构先验初始化与 RGB-Depth 双分支联合扩散引导，结合退火负提示策略与尺寸条件剪枝，在单卡 A100 约 1 小时内生成具备细粒度几何与逼真外观的 3D 人体。

## 研究问题与动机
- 现有文本驱动 3D 人体方法多基于网格或 NeRF，配合 SDS 优化时，网格难以刻画衣物褶皱与配饰等细粒度拓扑，而 NeRF 渲染与训练耗时耗显存，难以兼顾效率与质量。
- 3D Gaussian Splatting (3DGS) 渲染高效，但原生训练依赖密集多视图初始化；在纯文本驱动设置下，稀疏梯度难以稳定优化点基表示的几何与外观。
- 原生 SDS 为补偿文本对齐需极大 CFG 尺度（如 100），导致生成图像过饱和或卡通化；同时 SDS 损失的高方差会使 3DGS 自适应密度控制不稳定，产生模糊几何与漂浮伪影。
- 如何将显式 3DGS 高效适配至强结构约束、细粒度要求高的人体生成领域，并解决初始化困难、梯度稀疏与伪影问题，是本工作的核心动机。

## 核心贡献（创新点）
1. **结构感知 SDS（Structure-Aware SDS）**：以 SMPL-X 网格表面采样初始化 3D 高斯，并扩展 Stable Diffusion 为 RGB-Depth 双分支联合去噪网络，利用多模态 score 同步引导高斯密度调节与几何/外观优化。**与仅依赖通用 SfM 点云或单一 RGB 扩散先验的方法本质不同，该方法锚定参数化人体拓扑并提供空间对齐的结构监督，从根本上缓解 3DGS 在文本驱动场景下的初始化失配与稀疏梯度问题。**
2. **退火负提示引导（Annealed Negative Prompt Guidance）**：将 SDS 解耦为生成得分与分类器得分，仅保留更干净的分类器得分，并引入按 timestep 衰减的负提示权重，在常规 CFG 尺度（7.5）下实现文本对齐。**与 DreamFusion 盲目放大 CFG 尺度以压制生成噪声的策略本质不同，该方法通过 cleaner classifier score 替代高方差 generative score，在低尺度下即可避免过饱和与风格失真。**
3. **尺寸条件高斯剪枝（Size-Conditioned Gaussian Pruning）**：在自适应密度控制后追加纯剪枝阶段，依据高斯缩放因子阈值移除低透明度大尺度实例，消除体表漂浮伪影。**与直接依赖 SDS 梯度波动进行密度增减的策略本质不同，该后处理剪枝机制利用几何冗余性安全清除视觉无效高斯，显著提升生成平滑度而不损失细粒度细节。**

## 方法详解
- **高斯初始化与姿态条件**：放弃 Shap-E/Point-E 等通用点云先验，直接在 SMPL-X 网格表面均匀采样 100k 个高斯作为初始中心（单位缩放、均值颜色、无旋转）。将 SMPL-X 关节映射为 COCO 风格 2D 骨骼图，剔除遮挡关节后作为 pose condition 拼接到扩散模型输入。
- **Texture-Structure Joint Model**：以 SD 2.0 为基座，复制 UNet 的 conv_in、首 DownBlock、末 UpBlock 与 conv_out 构建双头去噪网络，以 v-prediction 目标联合学习图像 RGB 与空间对齐深度图 d 的分布。损失函数为：
  $\mathbb{E}_{\mathbf{v}, t}[\|\mathbf{v}_\phi(\mathbf{x}_t; \mathbf{p}, y) - \mathbf{v}_t^\mathbf{x}\|^2 + \|\mathbf{v}_\phi(\mathbf{d}_t; \mathbf{p}, y) - \mathbf{v}_t^\mathbf{d}\|^2]$
- **Dual-Branch SDS 优化**：渲染时同步输出 RGB 图像 x 与按 $\alpha$-blending 累加的归一化深度图 d。优化梯度由双分支加权和构成：
  $\nabla_\theta \mathcal{L}_{SDS} = \lambda_1 \mathbb{E}[\cdots \frac{\partial \mathbf{x}}{\partial \theta}] + \lambda_2 \mathbb{E}[\cdots \frac{\partial \mathbf{d}}{\partial \theta}]$，其中 $\lambda_1=\lambda_2=0.5$。深度分支
