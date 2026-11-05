---
title: "Single-Mesh-Diffusion-Models-with-Field-Latents-for-Texture"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Mitchel_Single_Mesh_Diffusion_Models_with_Field_Latents_for_Texture_Generation_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 08:34:20"
field: "3D纹理生成"
keywords: ["texture generation", "diffusion models", "3D mesh", "field latents", "isometry-equivariant", "single-mesh generation"]
innovations: ["提出Field Latents将纹理编码为切向量场", "设计等距等变的Field Latent Diffusion Models", "支持单纹理网格上的生成性纹理迁移"]
benchmarks: ["Google Scanned Objects", "Objaverse"]
---

# 论文速读：Single-Mesh-Diffusion-Models-with-Field-Latents-for-Texture

## 一句话总结
本文提出了一种直接在三角网格表面上操作的内在扩散模型框架，通过引入**Field Latents（场潜在变量）**这一将纹理编码为切向量场的新表示，以及**Field Latent Diffusion Models（FLDMs）**实现高质量纹理生成与编辑，支持无条件生成、标签引导生成、图像修复和跨几何纹理迁移等任务。

## 研究问题与动机
- **现有方法的局限**：当前大多数3D纹理生成方法依赖2D图像的优化方式（如SDS分数蒸馏），存在视图不一致性和细节丢失问题；而基于三平面特征的方法需要将几何和纹理栅格化到3D网格，导致高频纹理细节因抗锯齿效应而丢失。
- **数据稀缺问题**：高质量的3D纹理数据集规模有限，且许多真实3D资产具有独特的几何或风格，难以在大模型上进行条件训练。
- **表面纹理生成的内在挑战**：网格表面点没有规范的方向，局部坐标系的选取存在任意旋转的不确定性，使得直接套用2D的VAE和扩散模型变得困难。
- **解决方向**：利用三角网格作为自然的离散表示空间，设计等距等变（isometry-equivariant）的模型以解决方向模糊问题，实现内在的表面扩散模型。

## 核心贡献（创新点）
1. **Field Latents（场潜在变量）**：提出一种新的潜在表示，将高分辨率纹理映射为网格顶点处的离散切向量场，相较于标量特征能够捕捉局部纹理的方向信息，实现有效的感知压缩。
2. **Field Latent Diffusion Models（FLDMs）**：设计了基于场卷积（field convolutions）的去噪扩散模型，学习切向量场上的扩散去噪过程，同时支持时间步和可选条件注入，保持等距等变性。
3. **单纹理网格范式下的灵活生成**：在单个纹理化网格上训练，可生成多样化的纹理变化，支持标签引导生成、图像修复、生成性纹理迁移等用户控制任务。
4. **等距等变性的理论保证与实践优势**：证明了FL-VAE和FLDM在等距变换下保持等变性，使得模型能够在局部相似区域无缝复现纹理细节，并实现跨几何的纹理迁移。
5. **预训练FL-VAE的通用性**：在平面网格上预训练的FL-VAE可以泛化到任意3D形状，无需针对每个网格单独训练编码器。

## 方法详解
**Field Latents（FL-VAE）**：
- **编码**：对于网格上每个顶点$p$，编码器$\mathcal{E}_p$将周围邻域（一阶环）内的纹理标量函数映射为$d$个独立正态分布的参数（均值$\mu_p^\psi$和标准差$\sigma_p^\psi$），其中均值是切向量（复数表示），标准差是标量。使用重参数化技巧采样潜在向量：$z_p^\psi = \mu_p^\psi + \sigma_p^\psi \odot \epsilon(p)$，其中$\epsilon \sim \mathcal{TN}_M(0, I_d)$是切空间上的多维高斯噪声。
- **解码**：解码器$\mathcal{D}_p$是一个神经场，利用对数映射$\log_p: B_p \to \mathbb{C}_p$构建局部坐标系，计算两个特征：(1) 不变标量特征——潜在代码与其共轭的外积的向量化；(2) 位置感知标量特征$c_{pq}^\psi = \log_p q \cdot \overline{z_p^\psi}$（存储切向量与相对位置的内积和行列式）。这两个特征拼接后输入5层MLP得到预测纹理。
- **等变性保证**：要求$d\gamma|_p \cdot \mu_p^\psi = \mu_{\gamma(p)}^{\gamma\psi}$且$\sigma_p^\psi = \sigma_{\gamma(p)}^{\gamma\psi}$，从而保证等距变换下的等变性。

**Field Latent Diffusion Models**：
- **前向过程**：在切向量场上逐步添加噪声：$Z_t = \sqrt{\alpha_t} Z + \sqrt{1-\alpha_t}\epsilon$，其中$\epsilon \sim \mathcal{TN}_M(0, I_d)$。
- **去噪网络**：基于场卷积（FC）构建浅层U-Net，卷积核为$\mathbf{f}_{c'c} \in L^2(\mathbb{C} \times \mathbb{R}^e, \mathbb{C})$，将时间步嵌入和可选条件嵌入直接注入卷积中，避免 additive/multiplicative embedding破坏等变性。
- **训练损失**：$L_{FLDM} = \mathbb{E}_{\epsilon, t}\|\epsilon - \varepsilon(Z_t, t, \rho)\|_2^2$
- **推理过程**：从完全噪声的向量场开始，迭代去噪，通过公式(11)逐步估计前一步的潜在向量场。

## 实验与结果
**数据集与基线**：
- 数据集：Google Scanned Objects、Objaverse
- 基线方法：INFs [27]、Sin3DM [58]、FL-VAE (Barycentric)

**纹理重建质量（Table 1）**：
- FL-VAE在高/低分辨率网格上均取得最优结果：PSNR 22.38/20.59，DSSIM 0.51/0.83，LPIPS 1.02/1.81，显著优于INFs和barycentric变体。

**无条件生成质量（Table 2）**：
- FLDM在SIFID上达到3.27（越低越好），显著优于Sin3DM的6.58；LPIPS为1.15 vs Sin3DM的2.20。
- Sin3DM的合成纹理出现细节重复或沿主轴外推的问题，且无法在不修改几何的情况下显著改变纹理。

**用户控制任务**：
- **标签引导生成**：通过用户绘制的语义标签，使生成纹理符合主观内容分布（如鞋子不同区域、骷髅的眼睛嘴巴）。
- **图像修复**：保留掩码区域的原始纹理，在外部合成新内容，边界处与原始纹理自然衔接。
- **生成性纹理迁移**：将在网格M上训练的FLDM采样到新网格N上，实现风格迁移，跨越不同拓扑（genus 0 vs genus 3）成功迁移。
- **纹理尺度控制**：通过调整采样网格分辨率控制合成纹理的细节大小。

## 相关工作脉络
- **基于2D图像的3D生成**：DreamFusion等使用SDS优化NeRF，TexFusion等在渲染图像上直接应用2D LDMs，但存在视图不一致和细节丢失问题。
- **三平面表征**：Sin3DM等方法将几何和纹理栅格化到3D网格后编码为三平面特征，高频细节因抗锯齿效应丢失，且纹理与3D嵌入强关联。
- **内在神经网络场（INFs）**：将纹理编码为Laplacian特征函数在顶点处的值，通过重心插值恢复面部特征，在高压缩比下重建质量下降。
- **流形扩散场**：基于全局注意力机制，计算复杂度限制其仅适用于低分辨率网格。
- **单图像生成**：SinGAN、SindDM、SinDiffusion等从单个图像学习内部patch分布，本文借鉴此范式到单个纹理网格上。
- **网格上GAN方法**：Texturify、Mesh2Tex将纹理压缩到网格facet并通过可微渲染进行对抗监督。

## 局限性与未来方向
- **训练时间较长**：FLDM的训练时间相对较长，可能限制实际应用。
- **无法表达方向性信息**：当前FLDM合成的纹理无法反映特定的方向信息，未来可通过让用户指定向量场条件、通过点积嵌入网络特征来解决。
- **依赖近等距几何**：纹理迁移要求源网格和目标网格近似等距，对于差异过大的形状效果可能受限。
- **单网格范式的扩展性**：目前仅在单个纹理网格上训练，扩展到多网格数据集可能面临新的挑战。

## 研究启发与可借鉴点
- **等距等变性的理论框架**：将等变约束引入扩散模型设计，解决了网格表面方向模糊的根本问题，这一思路可扩展到其他几何深度学习任务。
- **切向量场表示纹理**：用向量场而非标量场表征局部纹理，能够捕捉方向信息，在纹理压缩和生成中提供更丰富的表达。
- **对数坐标函数的优越性**：相比重心插值，利用对数映射构建的位置感知特征能更丰富地扩展顶点特征，提升重建质量。
- **预训练潜在表示的通用性**：在平面网格上预训练FL-VAE可泛化到任意3D形状，解耦了潜在表示学习与具体几何的关系。
- **场卷积的条件注入策略**：通过扩展卷积核维度而非additive/multiplicative embedding来注入条件，同时保持等变性，为其他等变扩散模型提供设计参考。

## 关键术语表
**Field Latents (FLs)**：将纹理编码为网格顶点处切向量场的新潜在表示，通过向量而非标量特征捕捉方向信息，实现有效的感知压缩。
**Field Latent Diffusion Models (FLDMs)**：在FL空间中学习去噪扩散过程的模型，基于场卷积构建等距等变的去噪网络。
**Isometry-equivariance**：模型在距离保持的形变（等距变换）下保持等变性质，解决网格表面局部坐标系的旋转模糊问题。
**Field Convolutions (FCs)**：作用于切向量场上的表面卷积算子，是构建等变网络的基础模块。
**Tangent vector features**：定义在网格顶点切空间中的向量特征，用复数表示，包含幅度和方向信息。
**Single-textured-mesh paradigm**：在单个纹理化网格上训练生成模型，生成该纹理的变化变体，避免大规模3D数据依赖。
**Generative texture transfer**：将在一个网格上训练的FLDM采样到另一个近似等距的网格上，实现风格的生成性纹理迁移。
**Logarithmic coordinate function**：利用对数映射$\log_p q$构建的位置感知特征，比重心插值更丰富地扩展顶点特征到网格面片。

## 关键超参数与实现细节
- **FL-VAE潜在维度**：$d = 8$
- **FL-VAE预训练**：Open-Images V4数据集，1K个含10K顶点的平面网格，每样本512×512图像
- **FLDM训练预处理**：每个网格创建500份副本（30K顶点），不同三角剖分防止过拟合连接结构
- **扩散时间步**：$T = 1000$，线性噪声调度
- **优化器**：Adam
- **去噪网络**：浅层两级U-Net，含场卷积

## 可复现要素
- **数据集**：Google Scanned Objects [13]、Objaverse [8]、Open-Images V4 [29]（公开）
- **代码**：已开源，网址 https://single-mesh-diffusion.github.io/
- **权重**：预训练FL-VAE和FLDM权重需从项目页面获取
- **关键超参**：见上方"关键超参数与实现细节"
