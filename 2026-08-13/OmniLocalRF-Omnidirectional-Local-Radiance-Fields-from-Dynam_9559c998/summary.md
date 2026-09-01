---
title: "OmniLocalRF-Omnidirectional-Local-Radiance-Fields-from-Dynam"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Choi_OmniLocalRF_Omnidirectional_Local_Radiance_Fields_from_Dynamic_Videos_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 15:19:04"
field: "360°全景神经辐射场与动态场景分解"
keywords: ["omnidirectional view synthesis", "dynamic object removal", "neural radiance fields", "LocalRF", "motion mask", "bidirectional optimization", "360 video"]
innovations: ["双向远帧优化机制：利用360°环形视场在相邻RF块间进行跨帧双向光度refinement", "无预训练的多分辨率等距特征平面运动掩码模块，在体积渲染中联合学习静动解耦", "在无需人工掩码与相机先验条件下实现长360°视频静态新视角合成与自校准位姿估计"]
benchmarks: ["self-collected Insta360 real 360° dataset (6 scenes)", "synthetic 360° dataset with floating dynamic objects (3 scenes)", "Mip-NeRF360, EgoNeRF, LocalRF, D²NeRF, RobustNeRF baselines"]
---

# 论文速读：OmniLocalRF-Omnidirectional-Local-Radiance-Fields-from-Dynam

## 一句话总结
本文提出 OmniLocalRF，从包含动态物体（含拍摄者）的 casual 360° 视频中新合成静态场景视图，通过双向远帧优化与多分辨率运动掩码联合训练，同时实现动态物体去除、缺失区域补全及自校准相机位姿估计。

## 研究问题与动机
1. 360° 全景相机拍摄时动态物体（含摄影师）不可避免进入视场，导致新视角合成产生重影伪影。
2. 现有 NeRF 方法（Mip-NeRF360、EgoNeRF）假设训练数据静态，无法直接处理含动态物体的长轨迹 360° 视频。
3. 基于时域建模的动态分离方法（D²NeRF、Neuraldiff）参数化成本高，难以扩展到长视频；RobustNeRF 依赖光度误差加权，在大场景下难以区分静动，且训练效率低。
4. LocalRF 可处理长轨迹但缺乏全局信息利用，对慢速移动物体敏感，易受动态物体产生的几何不一致干扰。

## 核心贡献（创新点）
1. **双向远帧优化机制**：在 LocalRF 渐进式窗口滑动基础上，利用 360° 视频的环形视场特性，将远帧射线投影到相邻已收敛 RF 块并反向监督，弥补单方向局部优化的全局信息不足。
2. **无预训练的运动掩码模块**：基于多分辨率等距柱状特征平面 + 轻量 MLP，直接在体积渲染射线上升采样预测动态颜色与运动掩码 alpha，无需外部分割模型或光流。
3. **联合静动解耦与位姿自校准**：在同一 Radiance Field 框架内同步优化静态 RF 块、掩码模块与相机位姿，使几何不一致区域被自动映射到掩码而不动摇静态结构优化。
4. **与已有工作的本质区别**：区别于时域 3D 时空建模（D²NeRF）或光度加权 outlier 抑制（RobustNeRF），本文在空间局部 RF 块之间利用 360° 环视重叠进行跨帧双向光度约束，更适合长 360° 视频且扩展性强。

## 方法详解
- **局部 Radiance Field 块分配**：沿视频时间轴滑动窗口 $\mathcal{W}_m$ 向相邻 RF 块 $\mathbf{RF}_{\Theta_m}$ 插入帧与位姿；当当前帧相机距块中心超出 contraction 范围时冻结该块并新建后续块，延续 LocalRF [26] 的大场景扩展策略。
- **Forward 步（远帧投影 refinement）**：以当前块 $\mathbf{RF}_{\Theta_c}$ 渲染深度 $\hat{D}_c$，通过相对位姿 $[R|t]_{w(c,i)\to w(p,j)}$ 与等距投影 $\Pi$ 将源像素投影至先前块 $\mathbf{RF}_{\Theta_p}$ 所用的远帧，得到目标像素 $\mathbf{p}_{\text{dst}}$；仅保留满足 $(1-\tau)\hat{D}_p \leq \hat{D}_c \leq (1+\tau)\hat{D}_p$ 且 $\hat{D}_c \leq 1$ 的有效射线 $\mathcal{R}_C$（$\tau=0.05$），在静态区域上计算 L1 光度损失 $\mathcal{L}_{\text{rgb,d}}^{\text{for}}$ 更新 $\mathbf{RF}_{\Theta_c}$ 与掩码。
- **Backward 步（防止过拟合远帧）**：反向将来自远帧的射线经 $\mathbf{RF}_{\Theta_p}$ 渲染深度后投影回近帧，用输入帧颜色对 $\mathbf{RF}_{\Theta_p}$ 进行光度监督 $\mathcal{L}_{\text{rgb,s}}^{\text{back}}$ 与 $\mathcal{L}_{\text{rgb,d}}^{\text{back}}$，避免只优化远帧导致相邻视角畸变。
- **运动掩码模块**：每帧维护一组 $L=4$ 层多分辨率等距特征平面 $\mathcal{Z}_k$（初始高 $h_k^0=128$，通道数 4）；沿射线在归一化 $(u,v)$ 上拼接多层特征后送入浅层 MLP $F_{\Theta_D}$ 输出动态颜色 $\hat{\mathbf{C}}^{\text{dy}}$ 与掩码 alpha $\hat{M}$；最终渲染颜色由 $\hat{\mathbf{C}}_m = \hat{M}\hat{\mathbf{C}}^{\text{dy}} + (1-\hat{M})\hat{\mathbf{C}}_m^{\text{st}}$ 合成。
- **掩码正则化与唯一性约束**：引入对动态颜色的噪声扰动监督 $\mathcal{L}_{\text{rgb}}^{\text{mask}}$ 避免 mask 承担细节；同时施加 TV 损失与 binary 损失 $\mathcal{L}_{\text{reg}}^{\text{mask}}$ 促使掩码趋向二值化并空间平滑。
- **渐进优化流程**：窗口内先以本地光度损失优化当前块与位姿；进入 refining 阶段后同时执行 forward/backward 双向优化并更新掩码；位姿优化辅以 LocalRF 的光流损失与单目深度归一化监督；最终视图合成在相邻块重叠区域按位置混合。

## 实验与结果
- **数据集**：自采 Insta360 户外真实 360° 视频 6 场景（5760×2880, 30fps），每场景取前 500 帧每隔 4 帧共 125 帧用于训练/测试（112 train / 13 test）；另生成 3 个含漂浮动态物体的合成 360° 视频各 125 帧评估位姿（ATE/RPE）。
- **评测指标**：PSNR、SSIM、LPIPS；等距加权 PSNR<sup>WS</sup>、SSIM<sup>WS</sup>；位姿 ATE、RPE<sub>r</sub>、RPE<sub>t</sub>。测试帧人工标注动态掩码，渲染后对动态区灰度填充再计算指标。
- **主要定量结果（真实集平均，Table 2）**：
  - Ours w/ pose：PSNR 27.72 / SSIM 0.8171 / LPIPS 0.3299
  - Ours wo/ pose：PSNR 27.73 / SSIM 0.8165 / LPIPS 0.3297
  - 对比 Mip-NeRF360 (26.88/0.8094/0.3585)、EgoNeRF (25.95/0.7609/0.4383)、LocalRF w/ pose (26.56/0.8041/0.3471)、D²NeRF (20.95/0.6105/0.5100)、RobustNeRF (20.78/0.7093/0.4864)。
- **合成集最强结果（Table 1）**：Ours wo/ pose 取得 PSNR 29.93 / SSIM 0.8648 / LPIPS 0.2610，接近静态-only Mip-NeRF360 (29.48/0.8625) 并显著优于其他含动态基线。
- **位姿估计（Table 3）**：Ours ATE 0.00165，优于 LocalRF 0.00376 并与静态 ground truth 0.00165 相当；RPE<sub>t</sub> 0.00081 最优。
- **结论**：在无需人工掩码、无需预估计位姿条件下，动态去除与新视角合成均优于现有方法，且位姿估计保持高精度。

## 相关工作脉络
1. **Mip-NeRF360 / EgoNeRF**：面向静态/unbounded 360° 场景的高质量渲染；本文定位为将其扩展到含动态物体的 casual 视频。
2. **LocalRF [26]**：渐进式局部 RF 块实现长轨迹自校准；本文继承其块分配思想，但通过双向远帧优化补充全局监督并抑制动态伪影。
3. **D²NeRF [54] / NeuralDiff [46]**：在时域 3D 联合参数化中分离动静；本文不依赖时空高维参数化，利用 360° 环形视场的空间重叠进行跨帧优化，更适配长视频。
4. **RobustNeRF [37]**：基于 IRLS 对 outlier 加权的动态抑制；本文指出其在大规模全景中光度残差难以可靠区分动静，改用显式掩码模块 + 双向约束。
5. **OmnimatteRF [24]**：需 Mask R-CNN + 光流提供粗掩码；本文掩码完全自学习、无预训练分割器依赖。
6. **OpenMVG [29] / OpenVSLAM [40]**：传统 SfM/SLAM 位姿估计；本文与之一致的是能在含动态物体数据中获得可比甚至更优位姿（合成集 ATE 与静态 ground truth 一致）。

## 局限性与未来方向
- 完全遮挡区域无法补全（受限于光度监督，NeRF 类方法共性局限）；可引入感知损失或生成模型（如 Stable Diffusion）进行语义补全。
- 等距柱状网格在极区存在过采样冗余，掩码预测效率不高；未来可改用均匀球面网格。
- 未引入全局束调整（bundle adjustment）与闭环检测，位姿精度仍局限于局部/半全局优化；可结合成熟 SLAM 后端增强鲁棒性。
- 训练耗时约 12 小时/125 帧（单张 A6000），对超长视频可扩展性仍待提升。

## 研究启发与可借鉴点
1. **双向远帧优化思想**可迁移至任何具有环形/重叠视场的局部场学习方法（如 Street View 序列、多车巡游重建），通过跨段监督提升全局一致性。
2. **多分辨率等距特征平面 + 轻量 MLP 的掩码头**无需外部分割模型即可在体积渲染框架内联合学习，可作为通用"静动解耦模块"嵌入其他 NeRF/3DGS pipeline。
3. **有效射线筛选策略**（深度容差区间 + 距离上限）在抑制动态伪影的同时保留几何可信区域，方法简洁且可复用于其他自监督 NeRF 位姿优化。
4. **消融设计**中对 backward 源帧光度监督（Eq.7）必要性的验证，提示在跨帧 refinement 中"目标块与被 refine 块必须互相监督"这一对称性原则具有普适价值。

## 关键术语表
**LocalRF**：将场景划分为多个局部 Radiance Field 块并沿相机轨迹渐进优化，以实现大尺度免先验位姿与新视角合成。
**Omnidirectional contraction**：将无限世界坐标映射到单位球内（$\|\mathbf{x}\|>1$ 时压缩到 $(2-1/\|\mathbf{x}\|)\hat{\mathbf{x}}$），使远端几何仍可被局部 RF 块编码。
**Equirectangular projection**：将球面 360° 图像展开为宽高比 2:1 的二维图像的标准投影方式，极区存在畸变与过采样。
**Motion mask alpha**：掩码模块输出的标量场 $\hat{M}(\mathbf{r})\in[0,1]$，表示射线处动态成分占比，与静态渲染颜色进行 Alpha 合成。
**Bidirectional optimization**：在 forward 步用远帧监督当前块、在 backward 步用近帧监督远帧所用块，双向共用静态重叠区域的光度误差。
**Valid ray set $\mathcal{R}_C$**：两 RF 块渲染深度相差不超过 $\tau=5\%$ 且当前块深度 $\leq 1$（处于 contraction 范围内）的射线集合，作为可信静态重叠区域。
**PSNR<sup>WS</sup> / SSIM<sup>WS</sup>**：对等距图像极区畸变进行球面均匀权重修正后的质量评价指标。
**ATE / RPE**：视觉里程计常用位姿精度指标，分别衡量绝对轨迹误差与相对平移/旋转误差。

## 可复现要素
- **数据集**：作者自采 6 场景 Insta360 真实 360° 视频及 3 场景合成视频；论文称代码开源供研究使用（https://link 1）。
- **代码/权重**：论文声明代码开源（footnote 1），具体仓库链接见原文；权重未提及。
- **关键超参**：特征平面层数 $L=4$，初始高 $h_k^0=128$，通道数 4；有效深度容差 $\tau=0.05$；训练硬件为单 NVIDIA A6000；真实集使用半分辨率输入，D²NeRF 因 48GB 显存限制使用 1/4 分辨率。
- **位姿先验**：基线方法使用 OpenVSLAM [40] 提供的预估计位姿；本文自校准实验（wo/ pose）从头估计位姿。
