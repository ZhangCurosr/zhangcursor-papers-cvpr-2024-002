---
title: "Improving-Training-Efficiency-of-Diffusion-Models-via-Multi"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_Improving_Training_Efficiency_of_Diffusion_Models_via_Multi-Stage_Framework_and_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 15:17:54"
field: "扩散模型高效训练"
keywords: ["Diffusion Models", "Training Efficiency", "Multi-Stage Framework", "Multi-Decoder U-Net", "Timestep Clustering", "Generative Modeling"]
innovations: ["提出共享编码器+多解码器U-Net架构，兼顾抗过拟合与参数适配", "设计基于最优去噪器功能距离的时间步聚类算法，自动划分训练阶段"]
benchmarks: ["CIFAR-10", "CelebA 32x32", "CelebA 256x256"]
---

# 论文速读：Improving-Training-Efficiency-of-Diffusion-Models-via-Multi

## 一句话总结
本文针对扩散模型训练与采样效率低的问题，提出了一种**多阶段框架（Multi-Stage Framework）**，通过设计**共享编码器+多解码器的U-Net架构**和**基于最优去噪器的时间步聚类算法**，实现了训练资源的高效分配与跨阶段干扰的有效缓解，在CIFAR-10、CelebA等数据集上显著提升了FID质量与训练效率（EDM训练FLOPs减少82%，LDM减少30%）。

---

## 研究问题与动机
- **核心问题**：扩散模型（Diffusion Models）在生成质量出色，但存在**训练缓慢**与**采样缓慢**两大瓶颈，源于需要跟踪大量前向/反向扩散轨迹，且需使用大参数量模型在多个时间步（噪声水平）上统一训练。
- **统一架构的不足**：在相同参数下，高噪声阶段（接近纯噪声，$t \to 1$）任务简单导致参数冗余（overparameterization），而低噪声阶段（接近真实数据，$t \to 0$）任务复杂导致欠拟合（underfitting）。
- **分离架构的不足**：各阶段完全独立训练会引发过拟合（overfitting），且缺乏跨阶段信息共享机制。
- **梯度不相似性**：不同噪声水平下的分布形状不同导致梯度方向差异大，统一梯度下降效率低下。

---

## 核心贡献（创新点）
1. **识别扩散模型效率低下的两大根源**：不同时间步所需的模型容量差异显著（高噪声阶段参数需求少，低噪声阶段需求多），以及不同时间步的梯度不相似性阻碍统一架构训练效率。与已有工作的本质区别：首次从"参数需求异质性"和"梯度不相似性"两个角度系统分析并量化。
2. **提出多阶段多解码器U-Net架构**：将一个全局共享编码器与多个针对各时间阶段定制的解码器结合，兼具统一架构的抗过拟合优势与分离架构的参数适配优势。与已有工作的本质区别：不同于DDPM/EDM的统一架构或Expert Denoiser类工作的完全分离架构，本文采用"共享+定制"的混合设计。
3. **设计基于最优去噪器的时间步聚类算法**：利用理论上的最优去噪函数$\epsilon_{\theta}^*(x,t)$计算功能距离，自动确定阶段划分边界$t_1, t_2$，使各阶段内回归任务尽可能同质化。与已有工作的本质区别：不同于Timestep-based/SNR-based/Gradient-based的启发式划分方法，本文从最优去噪器函数距离出发进行聚类。
4. **系统性实验验证高效性**：在CIFAR-10、CelebA等多数据集上与DPM-Solver、EDM、LDM等SOTA方法对比，证明FID提升与训练效率显著改善。

---

## 方法详解

### 4.1 多阶段U-Net架构设计
- 将整个时间区间$t \in [0,1]$划分为三个阶段：$[0, t_1), [t_1, t_2), [t_2, 1]$。
- **共享编码器（Shared Encoder）**：所有阶段共用一个编码器，类似标准U-Net结构，防止过拟合，并维持h-space（特征空间）在时间步间的一致性。
- **专用解码器（Separate Decoders）**：借鉴Mask-RCNN的多头结构，为每个阶段设计独立的解码器，通道维度按阶段定制：
  - Interval 0（靠近真实数据，任务复杂）：192通道
  - Interval 1（中间阶段）：128通道
  - Interval 2（靠近噪声，任务简单）：16通道
- LDM模型中编码器使用224通道，解码器分别为256、192、128通道。

### 4.2 基于最优去噪器的时间步聚类算法
- **最优去噪器定义（Proposition 1）**：对于扩散模型去噪器$\epsilon_\theta(x,t)$，最优解为：
  $$\epsilon_\theta^*(x;t) = \frac{1}{s_t \sigma_t}\left[x - s_t \frac{\sum_{i=1}^N \mathcal{N}(x; s_t y_i, s_t^2 \sigma_t^2 I) y_i}{\sum_{i=1}^N \mathcal{N}(x; s_t y_i, s_t^2 \sigma_t^2 I)}\right]$$
- **功能距离度量**：在给定$x_0, \epsilon$下，两时间步$t_a, t_b$的最优去噪器距离为：
  $$\mathcal{D}(\epsilon_{t_a}^*, \epsilon_{t_b}^*, x_0, \epsilon) = \frac{1}{n}\sum_{i=1}^n \mathbb{1}(|\epsilon_{t_a}^*(x_{t_a}) - \epsilon_{t_b}^*(x_{t_b})|_i \leq \eta)$$
- **功能相似度**：$S(\epsilon_{t_a}^*, \epsilon_{t_b}^*) = \mathbb{E}_{x_0, \epsilon}[\mathcal{D}(\cdot)]$
- **聚类优化目标**：找到最大的$t_1$使得$[0, t_1)$内平均相似度$\geq \alpha$，找到最小的$t_2$使得$[t_2, 1]$内平均相似度$\geq \alpha$。
- **算法流程**：采样$K$组$(y_k, \epsilon_k, t_k)$，计算各$t_k$与端点$t=0$和$t=1$的距离，通过阈值$\alpha$确定边界。

### 4.3 设计原理
- **共享编码器的合理性**：(i) 多任务学习中共享参数可缓解过拟合；(ii) 保持h-space跨时间步的一致性。
- **参数分配的直觉**：当$t \to 0$时最优去噪器是从数据分布到高斯分布的复杂映射（需大容量）；当$t \to 1$时为恒等映射（可小容量）。与Subspace Diffusion的思路一致。

---

## 实验与结果

### 数据集与评估
- **数据集**：CIFAR-10 (32×32)、CelebA (32×32)、CelebA (256×256)
- **评估指标**：FID（生成质量）、NFE（采样效率）、PFLOPs（训练效率）
- **基线方法**：DDPM、Score SDE、PFGM、DDIM、gDDIM、DEIS、DPM-Solver、EDM、LDM

### 生成质量（CIFAR-10）
| 方法 | NFE | FID ↓ |
|------|-----|-------|
| DPM-Solver | 20 | 2.73 |
| **Multistage DPM-Solver (Ours)** | 20 | **2.35** |
| EDM | 35 | 2.05 |
| **Multistage EDM (Ours)** | 35 | **1.96** |

- Multistage DPM-Solver仅需20 NFE即达到PFGM（147 NFE）同等FID（2.35）。

### 训练效率
| 数据集 | 方法 | 训练迭代 | GFLOPs | 总PFLOPs | FID |
|--------|------|---------|--------|---------|-----|
| CIFAR-10 | DPM-Solver | $4.5 \times 10^5$ | 17.65 | 7.94 | 2.73 |
| CIFAR-10 | **Multistage DPM-Solver** | $2.5 \times 10^5$ (**56%**) | 18.65 (106%) | **4.66 (59%)** | 2.71 |
| CelebA 32×32 | EDM | $5.7 \times 10^5$ | 17.65 | 10.06 | 1.55 |
| CelebA 32×32 | **Multistage EDM** | $4.3 \times 10^5$ (**75%**) | 19.25 (109%) | **8.28 (82%)** | 1.44 |
| CelebA 256×256 | LDM | $4.9 \times 10^5$ | 88.39 | 43.31 | 8.29 |
| CelebA 256×256 | **Multistage LDM** | $1.7 \times 10^5$ (**35%**) | 76.19 (86%) | **12.95 (30%)** | 8.38 |

- **最强结果**：LDM训练PFLOPs从43.31降至12.95，**减少70%（节省约70%计算量）**；EDM训练FLOPs减少82%。

### 架构对比消融（CIFAR-10）
| 方法 | GFLOPs | FID ↓ |
|------|--------|-------|
| Unified | 17.65 | 2.73 |
| Separate | 17.65 | 2.87 |
| Separate (+ES+TP) | 18.65 | 2.52 |
| **Multistage (Ours)** | 18.65 | **2.35** |

### 聚类方法对比
| 聚类方法 | $t_1$ | $t_2$ | FID ↓ |
|---------|-------|-------|-------|
| Timestep | 0.330 | 0.670 | 3.12 |
| SNR | 0.348 | 0.709 | 2.72 |
| Gradient | 0.360 | 0.660 | 2.75 |
| **Optimal Denoiser (Ours)** | 0.442 | 0.631 | **2.35** |

---

## 相关工作脉络
1. **Choi et al. [19]（Perception Prioritized Training）**：重新配置不同时间区间的损失权重以提升性能，但未改变架构设计。
2. **Go et al. [20, 22]（Multi-task Learning for Diffusion）**：将不同区间视为不同任务，采用多任务学习策略，使用梯度相似性进行聚类；本文采用更理论化的最优去噪器距离。
3. **Lee et al. [21]（Multi-Architecture Multi-Expert DM）**：基于频率特性为各区间设计独立架构；本文通过共享编码器克服其过拟合缺陷。
4. **Balaji et al. [28]（Ediff-I）**：文本到图像扩散模型中使用独立专家去噪器；本文方法通用性更强，无需文本条件。
5. **Lu et al. [24]（DPM-Solver）**与**Karras et al. [18]（EDM）**：高效采样求解器；本文框架可与这些采样器无缝集成，进一步提升整体效率。

---

## 局限性与未来方向
- **局限性**：目前仅在无条件图像生成上验证，尚未在条件生成（如文本引导）或逆向问题求解中测试。
- **未来方向**：(i) 扩展至条件扩散模型（Conditional DMs）；(ii) 应用于基于扩散模型的逆问题求解器（Inverse Problem Solvers）；(iii) 应用于大规模Stable Diffusion训练，预计可大幅降低计算需求。

---

## 研究启发与可借鉴点
1. **时间步分区思想的普适性**：将扩散过程分段训练的范式可迁移至其他序列生成任务（如视频扩散、3D生成），值得探索。
2. **共享+定制混合架构**："全局共享编码器 + 阶段专用解码器"的设计兼顾了参数效率与泛化能力，可作为类似架构设计的参考模板。
3. **基于理论最优解的聚类方法**：利用最优去噪器的函数距离进行时间步聚类，比启发式阈值法更具理论依据，可推广至其他需分段处理的生成模型训练。
4. **FLOPs与迭代次数的联合优化**：本文同时报告GFLOPs、训练迭代次数与PFLOPs，为效率评估提供了多维度的量化基准。
5. **可与本团队方向结合**：若团队研究条件扩散模型或高效采样，可将多阶段框架与现有方法（如DPM-Solver、一致性模型）结合，探索进一步加速空间。

---

## 关键术语表
- **Diffusion Models（扩散模型）**：通过逐步添加噪声训练、再学习反向去噪过程来生成数据的生成模型。
- **Denoising Score Matching（去噪分数匹配）**：扩散模型的核心训练技术，通过学习加噪数据的梯度（score function）来逼近数据分布。
- **FID（Frechet Inception Distance）**：衡量生成图像质量的指标，值越低表示生成质量越好。
- **NFE（Number of Function Evaluations）**：采样过程中神经网络前向传播的次数，衡量采样效率。
- **PFLOPs（Peta-Floating Point Operations）**：总训练计算量单位，用于衡量训练效率。
- **h-space**：U-Net中编码器输出的隐层特征空间，具有跨时间步的一致性和语义可操作性。
- **Optimal Denoiser（最优去噪器）**：给定数据分布下，去噪任务在理论上的最优解，用于时间步聚类分析。

---

## 可复现要素
- **数据集**：CIFAR-10、CelebA 32×32、CelebA 256×256（均为公开数据集）
- **代码开源情况**：论文未提及代码开源
- **权重开源情况**：论文未提及
- **关键超参数**：$\alpha$（相似度阈值）、$\eta$（距离判定阈值）、$K$（采样数量）、分阶段通道数（192/128/16或256/192/128）；具体数值论文未明确给出，需在附录或代码中确认。

---
