---
title: "GoMAvatar-Efficient-Animatable-Human-Modeling-from-Monocular"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Wen_GoMAvatar_Efficient_Animatable_Human_Modeling_from_Monocular_Video_Using_Gaussians-on-Mesh_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 12:22:42"
field: "神经渲染与数字人建模"
keywords: ["数字人重建", "单目视频", "Gaussian Splatting", "可变形网格", "实时渲染", "动画化avatar"]
innovations: ["提出Gaussians-on-Mesh混合表示，将3D Gaussian绑定于三角网格面实现显式正向形变", "设计伪漫反射/伪阴影双通道分解渲染，兼顾外观质量与视角相关光照", "端到端可微分训练GoM，兼容OpenGL图形管线，内存仅3.63MB/主体"]
benchmarks: ["ZJU-MoCap", "PeopleSnapshot"]
---

# 论文速读：GoMAvatar-Efficient-Animatable-Human-Modeling-from-Monocular

## 一句话总结
提出 GoMAvatar，一种从单目 RGB 视频高效构建可动画化数字人的框架，核心创新是 Gaussians-on-Mesh（GoM）混合表示，结合 3D Gaussian 的高渲染质量与可变形网格的显式几何优势，实现实时（43 FPS）自由视角渲染与低内存占用（3.63 MB/主体）。

## 研究问题与动机
- **核心问题**：如何从单目视频高效重建高质量、可动画化、支持自由视角渲染的数字人体，同时兼顾渲染速度、内存效率与图形管线兼容性。
- **现有方法不足**：
  - NeRF 类隐式表示方法（如 HumanNeRF、MonoHuman）渲染质量高但推理极慢（秒级），且缺乏显式几何，难以与游戏引擎集成；
  - 纯网格方法易出现自穿透、"幽灵效应"和体素插值模糊；
  - 原始 3D Gaussian 缺乏显式拓扑约束，在复杂姿态下形变不规则，无法直接用于正向运动学动画。

## 核心贡献（创新点）
1. **提出 Gaussians-on-Mesh（GoM）混合表示**：将 Gaussian splatting 与可变形网格绑定，兼顾 Gaussian 的高渲染质量与网格的显式几何/拓扑约束。
2. **设计局部坐标系 Gaussian 参数化与正向变形管线**：在三角形局部坐标中学习 Gaussian 旋转/缩放，通过面顶点变换到世界坐标，使 Gaussian 形状随姿态自适应，替代不可逆的隐式反向映射。
3. **引入伪漫反射/伪阴影双通道分解渲染**：用 Gaussian splatting 渲染伪漫反射图，用网格法线图经轻量 CNN 预测伪阴影图，二者相乘得到最终图像，提升视角相关光照建模能力。
4. **端到端可微分训练且兼容 OpenGL 图形管线**：无需烘焙或缓存两步流程，GoM 一次性学习，支持标准栅格化渲染，模型仅 3.63 MB。

## 方法详解
**1. GoM 表示**（§3.1）
- 规范空间中由顶点集合 $\{v^c\}$ 与三角面集合 $\{f_j\}$ 组成。
- 顶点 $v^c_i = (p^c_i, w_i)$，含位置与 LBS 权重；面 $f_j = (r_j, s_j, c_j, \{\Delta_{j,k}\})$，含局部 Gaussian 旋转、缩放、颜色及所属三顶点索引。
- 初始化：旋转为零、尺度为 1，使 Gaussian 在法线方向变薄，投影椭圆近似 Steiner 椭圆。

**2. 渲染管线**（§3.2）
- 颜色分解：$I = I_{GS} \cdot S$，其中 $I_{GS}$ 为 Gaussian splatting 渲染的伪漫反射图，$S$ 为由网格法线图经位置编码 + 1×1 CNN 预测的伪阴影缩放因子。
- Gaussian 世界坐标构建：均值 $\mu_j$ 为三角形面质心；协方差 $\Sigma_j = A_j(R_j S_j S_j^T R_j^T)A_j^T$，$A_j$ 为局部→世界变换矩阵，使 Gaussian 随面变形而自适应。

**3. 正向运动学变形**（§3.3）
- 非刚性变形：$p^{nr}_i = p^c_i + \text{NRDeformer}_\theta(\gamma(p^c_i), P)$。
- LBS 正向融合：$p^o_i = \text{LBS}(p^{nr}_i, w_i, P)$，避免隐式方法的反向映射歧义。

**4. 姿态精化**（§3.4）
- 用 PoseRefiner 网络对估计姿态 $\hat{P}$ 补充旋转修正 $\xi_j \in SO(3)$，仅在训练与新视角合成时使用，不用于动画。

**5. 训练损失**（§3.5）
- $L = L_I + \alpha_{lpips}L_{lpips} + \alpha_M L_M + \alpha_{reg}L_{reg}$
- $L_{reg} = L_{mask} + \alpha_{lap}L_{lap} + \alpha_{normal}L_{normal} + \alpha_{color}L_{color}$，分别约束轮廓、拉普拉斯平滑、法向一致与相邻面颜色平滑。
- 训练时使用 GoM 细分策略：沿边中点插入新顶点，每面拆为四面，属性复制，提升几何细节。

## 实验与结果
- **数据集**：ZJU-MoCap（6 个受试者）、PeopleSnapshot（4 个受试者）、YouTube 舞蹈视频。
- **基线**：Neural Body、HumanNeRF、NeuMan、MonoHuman、Anim-NeRF、InstantAvatar。
- **ZJU-MoCap 新视角**：PSNR 30.37 dB / SSIM 0.9689 / LPIPS* 32.53；**新姿态**：PSNR 30.34 dB / SSIM 0.9688 / LPIPS* 32.39，与最强基线 MonoHuman 相当。
- **PeopleSnapshot 新视角**：PSNR 30.68 / SSIM 0.9767 / LPIPS 0.0213，显著优于 InstantAvatar（28.61 / 0.9698 / 0.0242）。
- **速度**：23.2 ms/帧（A100 GPU）= **43 FPS**，比 MonoHuman 快 257×、比 HumanNeRF 快 76×、比 Anim-NeRF 快 8.4×。
- **内存**：3.63 MB，仅次于 NeuMan（2.27 MB），远小于 HumanNeRF（245.73 MB）、MonoHuman（280.67 MB）。
- **几何**：NC 0.6201 为所有方法最高；CD 2.8364 略低于 MonoHuman（2.6303），归因于 Gaussian 厚度导致掩码略大。

## 相关工作脉络
1. **HumanNeRF [70]**：NeRF + LBS 反向映射建数字人，渲染质量高但推理慢（1.77 s/帧），存在自穿透与浮点伪影；GoMAvatar 用正向 Mesh 形变替代反向映射，速度提升 76×。
2. **MonoHuman [81]**：基于 TSDF/隐式场的人体重建，性能最强但推理 5.97 s/帧、模型 280 MB；GoMAvatar 以 GoM 显式表示实现同等质量同时大幅压缩内存与延迟。
3. **InstantAvatar [26]**：60 秒快速重建单目人体，PSNR 28.61；GoMAvatar 在 PeopleSnapshot 上 PSNR 提升 2 dB+ 且速度提升近 3×。
4. **Dynamic 3D Gaussians [45] / 4D Gaussian Splatting [72]**：将 Gaussian 用于动态场景，但缺乏显式拓扑与正向动画接口；GoMAvatar 将 Gaussian 绑定于网格面，实现可驱动正向运动学的显式表示。
5. **NeuMan [27]**：极小内存（2.27 MB）但渲染质量较低（PSNR 28.96）；GoMAvatar 在相似量级内存下实现更高 PSNR（30.37）。
6. **MobileNeRF [13] / PlenOctrees [79]**：用烘焙/缓存加速 NeRF，需两步训练；GoMAvatar 端到端训练 GoM，避免二次烘焙导致的画质损失。

## 局限性与未来方向
- **几何精度受 Gaussian 厚度限制**：Gaussian 沿法线方向有一定厚度，导致重建网格略小于真实几何（CD 略高）。
- **非刚性变形依赖 MLP**：NRDeformer 为轻量 MLP，对极端肌肉形变或厚衣物拉伸可能建模不足。
- **姿态精化仅适用于训练阶段**：对新视频需重新精化，无法完全脱离初始姿态估计质量。
- **自穿透处理依赖显式网格拓扑**：对于严重自交叠姿态（如双手交叉于胸前）仍有几何歧义，尚未引入拓扑自适应机制。
- **未来方向**：引入拓扑可变网格或神经隐式符号距离场辅助几何完善；结合物理驱动的肌肉/衣物模拟扩展至更复杂穿着；探索与神经辐射场轻量化烘焙的混合加速。

## 研究启发与可借鉴点
1. **局部坐标系 Gaussian 参数化**：将 Gaussian 旋转/缩放绑定于三角面局部系，通过面顶点变换到世界系，使 Gaussian 形状随姿态自适应——该思想可迁移至任意网格驱动的点云/粒子渲染任务。
2. **伪漫反射/伪阴影解耦**：用 Gaussian 负责外观、用网格法线 + 轻量 CNN 负责光照调制，兼顾渲染质量与视角相关效果，可推广至一般物体的高效神经渲染。
3. **正向 LBS 变形 + 非刚性残差**：用显式正向运动学替代隐式反向映射，避免 ill-posed 问题，该设计可用于其他基于骨骼驱动的 3D 表示动画系统。
4. **GoM 细分策略**：训练中插入边中点顶点并复制面属性以提升细节，是一种轻量级多分辨率细化手段，适用于网格+点混合表示的联合优化。

## 关键术语表
**Gaussians-on-Mesh（GoM）**：将 3D Gaussian 绑定于三角网格面的混合表示，兼顾 Gaussian 的快速渲染与网格的显式几何/拓扑约束。  
**Linear Blend Skinning（LBS）**：基于骨骼权重的正向形变方法，用于将规范空间顶点映射到观测空间。  
**Non-rigid Deformer（NRDeformer）**：预测姿态相关非刚性偏移的 MLP，补偿 LBS 无法表达的肌肉/软组织形变。  
**Pseudo Albedo Map**：由 Gaussian splatting 直接渲染的表面颜色图，不含视角相关光照效应。  
**Pseudo Shading Map**：由网格法线图经轻量 CNN 预测的视角相关光照缩放图，与伪漫反射相乘得最终颜色。  
**Steiner Ellipse**：三角形内切椭圆中面积最大的椭圆，GoM 初始化时 Gaussian 投影近似该椭圆以获得贴合面的形状。  
**LPIPS\***：LPIPS 乘以 1000，用于论文中的数值比较（越低越好）。  
**SoftRasterizer**：可微分网格光栅化器，用于训练时生成软掩码与法线图。

## 可复现要素
- **数据集**：ZJU-MoCap（公开）、PeopleSnapshot（公开）、YouTube 视频（自行下载）。
- **代码/权重**：论文未明确声明开源链接，主页 https://wenj.github.io/GoMAvatar/ 可能提供代码；模型文件大小 3.63 MB/主体。
- **关键超参**：$\alpha_{lpips}$、$\alpha_M$、$\alpha_{reg}$、$\alpha_{lap}$、$\alpha_{normal}$、$\alpha_{color}$（论文表格中未列出具体数值，见附录或代码）；训练包含 GoM 细分步骤。
- **推理环境**：NVIDIA A100 GPU，43 FPS（23.2 ms/帧）。
