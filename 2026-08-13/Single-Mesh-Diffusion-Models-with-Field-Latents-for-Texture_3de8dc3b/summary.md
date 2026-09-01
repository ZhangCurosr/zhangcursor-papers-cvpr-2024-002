---
title: "Single-Mesh-Diffusion-Models-with-Field-Latents-for-Texture"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Mitchel_Single_Mesh_Diffusion_Models_with_Field_Latents_for_Texture_Generation_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 20:15:43"
field: "3D生成与纹理合成"
keywords: ["texture generation", "diffusion model", "field latent", "isometry equivariance", "single-mesh", "manifold deep learning", "3D shape"]
innovations: ["Field Latents将纹理编码为顶点切向量场以实现更高保真压缩", "FLDM在等距等变约束下学习曲面切空间扩散去噪", "单纹理网格范式下支持标签引导、inpainting与跨几何生成式纹理转移"]
benchmarks: ["Google Scanned Objects", "Objaverse"]
---

# 论文速读：Single-Mesh-Diffusion-Models-with-Field-Latents-for-Texture

## 一句话总结
本文提出一种操作在3D网格曲面内在的潜变量扩散模型框架（FLDM），通过将纹理编码为顶点处的切向量场（Field Latents），在单个带纹理网格上训练出等距等变（isometry-equivariant）的生成模型，实现高质量纹理合成、编辑与跨几何的生成式纹理转移。

## 研究问题与动机
- 大多数3D纹理合成方法依赖2D图像域（如SDS优化或三平面latent），将3D表示投影到图像再渲染回3D，导致视图不一致性与高频细节混叠
- 大规模高质量带纹理3D数据集稀缺，仅少数类别含复杂非均匀纹理；且大量真实资产几何/风格独特，难以直接条件化大规模模型
- 在纹理合成优先级高于几何重建的应用中，需避免为适配表征而牺牲保真度
- 曲面点无全局规范朝向，难以直接套用图像域VAE/DM；需设计对等距变换等变的表示以解决局部坐标歧义

## 核心贡献（创新点）
- **提出Field Latents（场潜变量）表示**：将高分辨纹理压缩为定义在低分辨率三角网格顶点上的离散切向量场；与INF的标量/Laplacian特征不同，切向量捕获局部方向信息，支持更高压缩比下的细节重建
- **提出Field Latent Diffusion Model（FLDM）**：在切向量潜空间上构建等距等变的扩散去噪模型，使用场卷积（FC）作为局部支撑算子；与Sin3DM等外在方案相比，避免将纹理与3D嵌入强耦合
- **建立单纹理网格范式下的可编辑生成流程**：仅需一个带纹理网格即可训练，支持无条件生成、标签引导生成与inpainting；并通过等距等变性实现跨拓扑/相似几何的生成式纹理转移

## 方法详解
- **Field Latents（FL-VAE）**
  - 顶点处切空间用局部正交基底表示为复数 $v \equiv re^{i\theta}$；噪声定义为切丛上的多维高斯 $\epsilon(p)=\epsilon_1+i\epsilon_2,\ \epsilon_{1,2}\sim\mathcal{N}(0,I_d)$
  - 编码器：在顶点$p$的一邻域（one-ring）内采样固定数量点，将纹理标量值与对数映射坐标 $\log_p q_i$ 交织为切向量特征，经8层VN-Transformer预测均值 $\mu_p^\psi\in\mathbb{C}_p^d$ 与标准差 $\sigma_p^\psi\in\mathbb{R}_{\ge0}^d$；重参数化得 $z_p^\psi=\mu_p^\psi+\sigma_p^\psi\odot\epsilon(p)$
  - 解码器为5层MLP神经场：对查询点$q$，构造不变特征 $\mathrm{vec}_{j\ge i}(z_p^\psi[z_p^\psi]^*)$ 与位置感知特征 $c_{pq}^\psi=\log_p q\cdot\overline{z_p^\psi}$；在每个三角面内由三个相邻顶点的预测经重心插值融合（仅在推理时）
  - 等距等变性充分条件：$d\gamma|_p\cdot\mu_p^\psi=\mu_{\gamma(p)}^{\gamma\psi}$ 且 $\sigma_p^\psi=\sigma_{\gamma(p)}^{\gamma\psi}$
- **Field Latent Diffusion Model（FLDM）**
  - 前向加噪：$Z_t=\sqrt{\alpha_t}Z+\sqrt{1-\alpha_t}\epsilon$，$\epsilon\sim\mathcal{TN}_M(0,I_d)$
  - 反向去噪网络 $\varepsilon(Z_t,t,\rho):TM^d\times\mathbb{R}_{\ge0}\times L^2(M,\mathbb{R}^m)\to TM^d$，训练损失为 $\|\epsilon-\varepsilon(Z_t,t,\rho)\|_2^2$
  - 条件注入：扩展FC卷积核为 $\mathbf{f}_{c'c}\in L^2(\mathbb{C}\times\mathbb{R}^e,\mathbb{C})$，避免加法嵌入破坏等变性、乘法嵌入不稳定
  - 网络结构为浅层两级U-Net（仿SinDiffusion），感受野受限以防在单样本上过拟合
- **训练协议**
  - FL-VAE预训练：在OpenImagesV4上，将$512\times512$图像叠加随机三角剖分（1K种、每种10K顶点），d=8；强制用粗网格重建高频提升鲁棒性
  - FLDM训练：对目标网格复制500份（每份30K顶点、不同剖分、共享UV），T=1000线性噪声调度；防止网络记忆连通性

## 实验与结果
- **重建（Table 1）**：Google Scanned Objects，30K/5K顶点；FL-VAE在PSNR/DSSIM/LPIPS上全面优于FL-VAE(Bary.)与INFs，证明对数坐标函数比重心插值更丰富
- **无条件生成（Table 2）**：Objaverse + Scanned Objects，64张样本/24视角；FLDM SIFID=3.27、LPIPS=1.15，显著优于Sin3DM（6.58 / 2.20）；Sin3DM因合成几何会虚增LPIPS
- **可视化结论**：FLDM可在局部相似区域无缝复现纹理细节；Sin3DM因外在表示导致纹理沿主轴重复/外推，且难以在不改几何的情况下改变纹理
- **可编辑性**：标签引导生成（Figure 4）与inpainting（Figure 5，掩码边界自然衔接）均可行
- **生成式纹理转移（Figure 6）**：在壳类mesh上训练标签条件FLDM，采样到新壳/跨 genus 的骷髅mesh仍可稳定生成风格一致纹理
- **尺度控制（Figure 7）**：在同一模型上于不同分辨率网格采样，随顶点数减少细节尺度增大

## 相关工作脉络
- **DreamFusion/SDS系列**：外在地用预训练2D LDM做优化先验；计算昂贵且难以生成高频细节
- **Sin3DM**：首个单纹理网格范式的方法，但将几何+纹理栅格化到3D网格再三平面编码，高频细节易混叠且纹理与空间绑定
- **Intrinsic Neural Fields (INF)**：顶点处Laplacian特征+barycentric插值；在粗压缩下细节破碎，本文FL-VAE以向量场+对数坐标超越
- **Manifold Diffusion Fields / Diffusion Probabilistic Fields**：全局注意力，仅适用于粗网格；本文用局部FC扩展至高分辨
- **SinGAN/SinDiffusion**：单图像扩散思路的本工作的生成范式来源；本文将其从图像推广到曲面切向量场并引入等变约束
- **Texturify/Mesh2Tex**：GAN+可微渲染的监督；本文在单网格扩散范式下获得更高保真度与可控编辑能力

## 局限性与未来方向
- 单网格FLDM训练时长较长
- 当前生成纹理仅反映标量分布，未显式建模/复用文本中的方向信息；作者提议未来可引入用户指定向量场作为条件（通过点积嵌入）
- 等距等变假设在局部曲率显著变化或拓扑远离时仍可能限制跨域匹配质量（虽示例显示对 genus 差异有容忍度）
- 当前解码仅在推理时重心插值，训练期隐式依赖顶点特征分布

## 研究启发与可借鉴点
- **切向量潜表示替代标量潜表示**：在曲面纹理/方向敏感任务中，用复切向量捕获局部结构比标量更富有表达力，可作为通用表征设计参考
- **对数映射坐标函数构造神经场插值**：$\log_p q\cdot\overline{z_p^\psi}$ 比barycentric更能保留局部几何语义，适合任何需在流形上延伸顶点特征的场景
- **等距等变扩散的条件注入范式**：用扩展核 $L^2(\mathbb{C}\times\mathbb{R}^e,\mathbb{C})$ 而非加法/乘法嵌入，可同时满足稳定性与对称性；可扩展到其他流形扩散任务
- **单样本预训练+重剖分增强**：在粗三角网上预训练VAE、对目标网格做多次随机重剖分的data augmentation策略，兼顾高频学习与防过拟合，可直接迁移到网格/点云扩散任务
- **生成式纹理转移的等变利用**：等距等变不仅是正则，更是"不同几何上的同一潜空间"的充分条件；可在跨形态生成、形状编辑等场景中复用该思路

## 关键术语表
- **Field Latents (FL)**：将连续纹理编码为定义在网格顶点切空间中的离散向量场潜变量，实现感知压缩
- **FL-VAE**：在切向量场上学习的局部支撑变分自编码器，编码为多元高斯、解码为基于对数坐标的神经场
- **FLDM**：在FL潜空间上执行的等距等变扩散去噪模型，用于单纹理网格的条件/无条件生成
- **Isometry-equivariance**：模型对保距形变的交换性，使切向量潜表示不依赖局部坐标系选择，从而支持跨几何转移
- **Field Convolution (FC)**：作用在切向量场上的局部卷积算子，核定义在复数-实数乘积空间，支持等变条件注入
- **Single-textured-mesh paradigm**：仅以单个带纹理网格为训练源，学习其内部 patch 分布并生成多样化纹理的范式
- **Logarithmic coordinate function**：解码器中利用 $\log_p q$ 构造的位置感知特征，较barycentric插值提供更丰富的面内表达
- **Generative texture transfer**：将在网格M上训练的FLDM直接采样到近似等距网格N上，实现风格化纹理迁移

## 可复现要素
- **数据集**：OpenImagesV4（VAE预训练）、Google Scanned Objects、Objaverse；论文代码与可视化公开在 https://single-mesh-diffusion.github.io/
- **代码/权重**：论文声明代码可用，具体开源仓库未在本段列出，需查看官网
- **关键超参**：FL-VAE latent维度 $d=8$；FLDM 最大时间步 $T=1000$，线性噪声调度；训练每网格 500 次重剖分副本（30K顶点）；优化器Adam
