---
title: "Semantic-Human-Mesh-Reconstruction-with-Textures"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Zhan_Semantic_Human_Mesh_Reconstruction_with_Textures_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 20:15:24"
field: "三维人体重建与纹理合成"
keywords: ["semantic human mesh", "texture generation", "non-rigid registration", "diffusion model", "UV completion", "monocular reconstruction"]
innovations: ["语义-法向采样(SNS)实现不完整输入鲁棒配准", "自监督UV域网格补全与细化网络", "文本驱动高分辨率语义纹理生成"]
benchmarks: ["THuman2.0", "CAPE-NFP"]
---

# 论文速读：Semantic-Human-Mesh-Reconstruction-with-Textures

## 一句话总结
本文提出 **SHERT**，一种可从详细3D表面（mesh或SDF）结合语义引导重建高质量语义人体网格的端到端管线，并支持从单目图像或文本驱动生成高保真纹理，解决现有方法在工业应用中面临的结果不稳定、网格质量低、缺乏UV展开与蒙皮权重等问题。

## 研究问题与动机
- 现有显式/隐式单目人体重建方法（如PIFu、ICON、ECON等）虽在几何细节上取得进展，但重建结果缺乏稳定的UV展开、一致语义信息与预定义蒙皮权重，难以直接用于工业下游（动画、换装、编辑）。
- 隐式方法在手部与面部重建上表现较差，且易产生不完整或几何不可分离的输出；显式方法难以适应灵活拓扑与细节捕捉。
- 现有非刚性配准方法主要面向精确对齐，面对真实场景中不完整、含噪声的目标表面时鲁棒性不足。
- 工业界对"可动画、可编辑、可更换部件"的语义人体Avatar需求强烈，需要统一支持面部/身体/手部的高质量语义网格与高分辨率纹理。

## 核心贡献（创新点）
- **语义-法向采样（SNS）**：利用SMPL-X提供的语义先验与顶点法向射线搜索目标表面，实现粗到细的非刚性配准；与现有方法本质区别在于显式支持不完整/错误输入的容错处理与面片剔除。
- **自监督网格补全网络**：将3D补全转化为UV域2D inpainting任务，结合随机掩码构造自监督训练对；与传统mesh completion算法本质区别在于保持顶点数量与相对位置一致的语义约束。
- **自监督网格细化网络**：利用图像与正反面法线图在UV域预测位移图优化细节（衣褶、姿态变形等）；与现有细化方法本质区别在于迭代可应用且与语义网格结构完全兼容。
- **文本驱动纹理扩散生成**：基于Stable Diffusion + ControlNet的微调，支持部分纹理inpainting与全局/局部文本驱动纹理重绘；相比DINAR等神经纹理方案本质区别在于分辨率高、UV稳定、支持文本控制。

## 方法详解
SHERT管线分为四个阶段：

**1) Subdivided SMPL-X 构建**  
以SMPL-X为基础（10,475顶点、54关节），通过中点细分算法两次细分，得到含149,921顶点、299,712面片的sub-SMPLX模型，兼顾表达能力与计算成本。

**2) 语义-法向采样（SNS）**  
从sub-SMPLX顶点沿法向发射射线与目标表面$\mathcal{M}_t$求交；对隐式场采用Ray Marching常数步长搜索。对采样三角面片计算三个质量指标：法向夹角$\theta$、面积比$s$、边长比$r$，超过阈值则剔除；再连通性检查移除小集合（默认$\theta=2, s=3, r=3, g=500$）。在规范空间重复上述操作以保证动画合理性，得到部分语义网格$\mathcal{M}_s$。

**3) 自监督网格补全**  
将$\mathcal{M}_s$映射至UV域，把3D补全转化为2D inpainting。引入随机掩码$\mathcal{H}_r$与真实缺失掩码$\mathcal{H}_o$构造训练对：

$$\mathcal{H}_{sup} = \mathcal{H}_r \cdot (1 - \mathcal{H}_o) + \mathcal{H}_o$$
$$\overline{d} = \frac{S_{sample} - S_{pose}}{N_{pose}}, \quad \overline{S} = S_{cano} + N_{cano} \cdot \overline{d}$$
$$\mathcal{L}_{inp} = \frac{\left\| (S_p - \overline{S}) \cdot (\mathcal{H}_{sup} - \mathcal{H}_o) \right\|_2^2}{\sum_{i,j}(\mathcal{H}_{sup} - \mathcal{H}_o)_{i,j}}$$

网络输出位移UV图$d$，补全后映射回姿态空间得到$\mathcal{M}_c$。手部/面部/足部等难区域可直接用sub-SMPLX对应部分替换，或接入EMOCA等FLAME-based方法进一步提升。

**4) 自监督网格细化**  
输入图像$I$、正反面法线图$\mathcal{N}_f, \mathcal{N}_b$与完整UV位置图$S_l$，通过U-Net提取图像域特征后投影至UV域：

$$F = \mathcal{P}(\mathcal{F}(I, \mathcal{N}_f, \mathcal{N}_b), S_l, c)$$
$$S_r = S_l + N_l \cdot z$$

损失函数：

$$\mathcal{L}_{dis} = MSE\left(z - \frac{S_c - S_l}{N_l}\right), \quad \mathcal{L}_{normal} = MSE(N_r - N_c)$$

支持迭代细化（图11展示2~3次迭代效果）。

**5) 文本驱动纹理生成**  
将重建语义网格投影到输入图像生成部分纹理图，基于Stable Diffusion + ControlNet微调，支持两种推理策略：①直接输入部分纹理图+随机噪声生成；②将编码后的部分纹理图与掩码拼接至潜特征以更好保留输入信息。使用DDIM采样器30步生成1024×1024纹理UV图，最终通过Real-ESRGAN进行超分。

## 实验与结果
- **数据集**：THuman2.0（训练499扫描、测试后27被试）、CAPE-NFP（100样本×3视角）、in-the-wild图像。
- **评估基线**：PIFu、PIFuHD、PaMIR、ICON、ECON、2K2K（单目重建）；N-ICP、RPTS、SVR-$\mathcal{L}_0$、Fast-RNRR、DR、MDA（配准）。
- **主要定量结果（Tab.1）**：在CAPE上，Ours(ICON-comp) P2S=0.8550↓、Chamfer=0.8107↓，优于ICON（0.8855/0.8609）；在THuman2.0上，Ours(ICON-refine) P2S=1.0442↓、Chamfer=1.0468↓，优于ECON-refine与ICON。
- **配准定量结果（Tab.2）**：SNS P2S=0.107↓、Chamfer=0.078↓、G-avg=0.729↑、$\theta<30°$=17.3↓，显著优于Fast-RNRR（P2S=0.115、G-avg=0.597）；含补全后完整网格27s（23s SNS + 4s completion）。
- **结论**：SHERT在几何保真度、网格质量与配准鲁棒性上均超越SOTA；定性结果（图9-12）显示面部、手部细节清晰，纹理质量高且支持文本驱动编辑。

## 相关工作脉络
- **PIFu/PIFuHD**：像素对齐隐式函数方法，预测顶点颜色而非可编辑纹理图，无法直接用于工业流程。
- **ICON/ECON**：显式/隐式结合的人体重建方法，SHERT在其输出基础上进一步提供语义网格与纹理生成，弥补下游可用性缺口。
- **DINAR**：基于神经纹理的diffusion inpainting方案，但与Avatar高度耦合且分辨率低；SHERT通过稳定UV展开实现高分辨率、文本驱动纹理生成。
- **非刚性配准（N-ICP/RPTS/MDA等）**：面向精确对齐设计，SHERT的SNS通过质量剔除与规范空间重采样提升对不完整/错误输入的鲁棒性。
- **Parametric/Implicit Clothing Models（SMPLicit等）**：专注宽松服装拟合，SHERT通过语义网格统一处理身体/面部/手部/服装细节。
- **Monocular Face Reconstruction（EMOCA等）**：SHERT保留与FLAME-based方法的接口，支持面部区域替换以提升精度。

## 局限性与未来方向
- SMPL-X几何先验限制导致对宽松衣物、鞋子、头发的重建弱于纯隐式方法。
- UV接缝处的纹理一致性难以保证（扩散模型在UV边界处可能产生不连续）。
- 文本驱动纹理生成受限于Stable Diffusion对UV参数化图像的训练分布，复杂姿态下生成质量可能波动。
- 未来方向：①扩展语义先验至更丰富服装/鞋履模型；②引入UV边界感知损失或post-processing提升接缝一致性；③结合视频序列实现时序一致的纹理生成。

## 研究启发与可借鉴点
- **自监督UV域补全策略**：将3D网格补全转化为2D图像inpainting并构造随机掩码训练对，思路可迁移至其他3D形状补全任务（物体、场景）。
- **规范空间+姿态空间双重视角**：SNS在规范空间重采样避免动画变形，该设计对任何依赖模板的先验配准方法均有参考价值。
- **扩散模型+ControlNet的纹理生成架构**：支持部分掩码+文本双条件的推理策略，可直接复用于产品材质生成、数字人换装等下游应用。
- **与下游SOTA重建器解耦**：SHERT以ICON/ECON为前置模块，表明"通用语义网格+纹理管线"可作为独立组件嵌入多种重建工作流。
- **迭代细化机制**：用户按需增加细化次数，平衡质量与延迟，适合交互式应用。

## 关键术语表
- **SHERT**：Semantic Human mEsh Reconstruction with Textures，本文提出的语义人体网格与纹理重建管线。
- **SNS（Semantic- and Normal-based Sampling）**：基于语义与法向的采样方法，沿SMPL-X顶点法向射线与目标表面求交获取部分语义网格。
- **sub-SMPLX**：对标准SMPL-X进行两次中点细分后的高密度模板网格（149,921顶点）。
- **UV域补全**：将3D网格补全转化为2D UV图像inpainting任务，利用自监督掩码训练。
- **Self-supervised Refinement Network**：在UV域预测位移图以细化网格几何细节的网络，使用拉普拉斯平滑网格作为输入。
- **ControlNet + Stable Diffusion**：本文用于纹理inpainting与文本驱动生成的扩散模型基础框架。
- **G-avg**：网格质量评价指标，衡量三角形面片质量的平均值。
- **LBS（Linear Blend Skinning）**：线性_blend蒙皮，用于将SMPL-X蒙皮权重迁移至重建网格实现动画。

## 可复现要素
- **数据集**：THuman2.0（公开）、CAPE（公开）、in-the-wild图像；训练使用THuman2.0前499扫描，测试使用后27被试。
- **代码**：论文未明确声明开源状态，项目主页 https://zhanxy.xyz/projects/shert 可查阅。
- **权重**：依赖ECON、ICON、Stable Diffusion、ControlNet、Real-ESRGAN等开源模型；自研网络训练100 epoch（lr=1e-6）与扩散模型1 epoch（lr=2e-5）。
- **关键超参**：SNS阈值$\theta=2, s=3, r=3, g=500$；UV分辨率1024×1024；扩散采样30步DDIM；三张NVIDIA RTX 3090 GPU训练。
