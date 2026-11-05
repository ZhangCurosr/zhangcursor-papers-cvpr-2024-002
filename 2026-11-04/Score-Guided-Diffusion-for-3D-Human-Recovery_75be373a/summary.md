---
title: "Score-Guided-Diffusion-for-3D-Human-Recovery"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Stathopoulos_Score-Guided_Diffusion_for_3D_Human_Recovery_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 16:38:16"
field: "3D human reconstruction"
keywords: ["3D human mesh recovery", "diffusion model", "score guidance", "inverse problem", "human pose estimation", "DDIM", "SMPL"]
innovations: ["将扩散模型作为学习先验并通过score guidance在潜空间迭代优化人体参数", "提出DDIM inversion-guided sampling循环统一求解单帧/多视角/视频三种HMR逆问题", "证明在统一框架下无需任务重训练即可超越传统优化基线（SMPLify/ProHMR-fitting）"]
benchmarks: ["3DPW", "EMDB 1", "Human3.6M", "Mannequin Challenge"]
---

# 论文速读：Score-Guided-Diffusion-for-3D-Human-Recovery

## 一句话总结
本文提出 ScoreHMR，将扩散模型作为 SMPL 人体参数的学习先验，通过 score guidance 在扩散潜空间迭代修正初始回归估计，以"数据驱动的迭代拟合"方式解决3D人体姿态与形状重建的逆问题，无需任务特定重训练即可适配单帧、多视角、视频序列等多种场景。

## 研究问题与动机
1. **传统优化方法缺陷**：SMPLify 等优化方案依赖手工设计能量项，存在多局部极小值、对初始化敏感、收敛慢等问题。
2. **纯回归方法不足**：即便如 HMR 2.0 等 SOTA 前馈网络能给出较好3D重建，但与图像观测（2D关键点等）的对齐仍存在明显偏差，缺少"回归+优化"协同中的可靠优化环节。
3. **扩散模型应用于人体恢复的空白**：扩散模型已在图像逆问题（超分、修复、去模糊）中作为结构先验使用，但尚未被用于3D人体姿态/形状估计的逆问题求解。
4. **需统一且灵活的框架**：希望用一个任务无关的扩散先验模型，配合不同 score guidance 项，即可在单帧拟合、多视角一致性、时序平滑等多个下游任务中通用，避免逐个任务重新训练。

## 核心贡献（创新点）
1. **Score-guided diffusion 作为学习先验优化器**：将 DDIM 去噪过程视为迭代优化，通过近似条件后验 score $\nabla \log p(\mathbf{x}_t|I,\mathbf{y}) \approx \nabla \log p(\mathbf{x}_t|I) + \nabla \log p(\mathbf{y}|\hat{\mathbf{x}}_0)$，在扩散潜空间中以梯度方式把人体模型推向与观测一致。*本质区别：不同于传统手工先验+能量最小化，这里用数据驱动的扩散先验替代 hand-crafted prior，并以 score guidance 替代显式优化步。*
2. **DDIM inversion–guided sampling 循环**：先把回归估计 $\mathbf{x}_{reg}$ 通过确定性 DDIM inversion 映射到噪声潜 $\mathbf{x}_\tau$，再沿 guided DDIM sampling 逐步去噪，循环直至对齐。*区别：以往工作仅在采样时使用固定步数，本文通过 loop 让对齐程度动态控制迭代，形成"隐式优化"。*
3. **统一框架下的三大应用**：(i) 单帧2D关键点重投影拟合；(ii) 多视角 cross-view consistency guidance；(iii) 视频序列 temporal consistency guidance（可选叠加关键点项）。*区别：三个任务共享同一个预训练扩散模型，仅需替换 $\mathcal{L}_{guide}$，无需任务微调。*
4. **实证表明能显著提升现有回归器**：在所有评测基准上均优于优化类 baseline（SMPLify、ProHMR-fitting、VIBE-opt），且是**唯一**能进一步提升 HMR 2.0 单帧 3D pose 性能（PA-MPJPE）的方法。

## 方法详解
- **问题形式化**：给定图像 $I$、初始回归 SMPL 参数 $\mathbf{x}_{reg}=\{\theta_{reg},\beta_{reg}\}$ 和观测 $\mathbf{y}$，目标是求 $\mathbf{x}_0$ 使 $\mathbf{y} = \mathcal{A}(\mathbf{x}_0)+\eta$ 成立。
- **扩散模型训练**：使用6D旋转表示，$\theta \in \mathbb{R}^{144}$；denoiser $\epsilon_\phi(\mathbf{x}_t,t,c)$ 由3个 MLP block 组成，每个 block 通过 sinusoidal timestep encoding 得到 $(\mathbf{t}_s,\mathbf{t}_b)$ 做 AdaLN，并与图像特征 $c=g(I)$ 拼接。图像 backbone $g$ 可冻结（实验中采用冻结，从预训练回归网络提取）。损失 $\mathcal{L}_{DM}=\mathbb{E}\|\epsilon_\phi(\mathbf{x}_t,t,I)-\epsilon\|^2$，伪 GT 来自 SPIN/EFT。
- **Score guidance 推导**：假设观测噪声高斯，近似后验 score：
  $$\nabla_{\mathbf{x}_t}\log p(\mathbf{y}|I,\mathbf{x}_t)\simeq -\rho\nabla_{\mathbf{x}_t}\|\mathbf{y}-\mathcal{A}(\hat{\mathbf{x}}_0(\mathbf{x}_t))\|^2$$
  其中 $\hat{\mathbf{x}}_0(\mathbf{x}_t)=\frac{1}{\sqrt{\alpha_t}}(\mathbf{x}_t-\sqrt{1-\alpha_t}\epsilon_\phi(\mathbf{x}_t,t,I))$。
- **Guided DDIM 更新**：修改噪声预测 $\epsilon'_\phi=\epsilon_\phi+\rho\sqrt{1-\alpha_t}\nabla_{\mathbf{x}_t}\|\mathbf{y}-\mathcal{A}(\hat{\mathbf{x}}_0)\|^2$，再按标准 DDIM 递推；循环直到 $\mathcal{L}_g$ 相对变化低于 $\lambda_{thr}$。
- **三种应用的 guidance loss**：
  1. **关键点对齐**：$\mathcal{L}_{repr}=\mathbf{y}_{conf}\|\Pi_K(W\mathcal{M}(\hat{\mathbf{x}}_0,\beta)+\gamma)-\mathbf{y}_{kp}\|^2$，同时优化相机平移 $\gamma$。
  2. **多视角一致性**：$\mathcal{L}_{MV}=\sum_n\|\hat{\mathbf{x}}_{0,b}^{(n)}-\bar{\mathbf{x}}_{0,b}\|^2$，约束各视角 body pose 一致。
  3. **时序一致性**：$\mathcal{L}_{temp}=\sum_{n\ge 2}\|\hat{\mathbf{x}}_0^{(n)}-\hat{\mathbf{x}}_0^{(n-1)}\|^2$，可选叠加 $\mathcal{L}_{repr}$。

## 实验与结果
- **数据集**：训练用 Human3.6M、MPI-INF-3DHP、COCO、MPII；评测 3DPW、EMDB 1、Human3.6M、Mannequin Challenge。
- **基线**：LGD、LFMM、SMPLify、ProHMR-fitting、VIBE-opt、VIBE、ProHMR、HMR 2.0。
- **单帧 model fitting（PA-MPJPE mm）**：
  - HMR 2.0 初始 54.3(3DPW)/78.7(EMDB1)；+ScoreHMR-b 降至 **51.1/76.6**，为最优。
  - 相较 ProHMR-fitting 对 HMR 2.0 无效（55.1 vs 55.1），ScoreHMR 是唯一提升 HMR 2.0 的方法。
  - ProHMR 初始 59.8；+ScoreHMR-b → **54.7**，优于 SMPLify(60.9) 与 ProHMR-fitting(55.1)。
- **多视角（MPJPE/PA-MPJPE mm）**：HMR 2.0 → +ScoreHMR-b：Human3.6M **44.7/29.0**，Mannequin **148.3/79.1**，全面优于 ProHMR-fitting。
- **视频时序**：HMR 2.0 → +ScoreHMR-b：3DPW PA-MPJPE **50.5**，Acc Err **11.1**；EMDB 1 PA-MPJPE **75.3**，Acc Err **11.9**。相对 ProHMR-fitting 加速度误差分别降低 **21.3%** 和 **40.5%**。
- **效率**：Mannequin 20K 帧仅 1.5 分钟；3DPW 35K 帧仅 14 分钟。
- **结论**：ScoreHMR 在全部三种任务、所有基准上 consistently 优于优化类 baseline；且对图像特征来源（ProHMR/PARE）与伪 GT（SPIN/EFT）具有鲁棒性。

## 相关工作脉络
1. **HMR 回归类**：HMR、PyMAF、PARE、HMR 2.0——本文直接以这些网络的输出作为起始估计，定位在后处理的 refine 层，而非替换回归器。
2. **HMR 优化类**：SMPLify、ProHMR-fitting、LGD、LFMM——传统优化依赖 hand-crafted prior 和能量最小化；ScoreHMR 用扩散 score 代替前项，避免手工调参。
3. **Diffusion for inverse problems**：Chung et al. (DPS)、Song et al. (PGDM) 等已在图像超分/修复中使用 score guidance；本文首次将其引入 3D 人体参数恢复。
4. **Diffusion for human motion**：Human Motion Diffusion Model、Single Motion Diffusion、PhysDiff 等用于生成 motion；本文关注"恢复/拟合"（inverse problem）而非生成。
5. **回归+优化协同**：ProHMR、Kolotouros et al. 等先用回归再优化；本文延续思路但把优化器替换为通用的 score-guided diffusion，扩展性更强。

## 局限性与未来方向
- **仅建模 pose 先验**：论文声明仅对 $\theta$ 训扩散模型，$\beta$ 保留原始回归结果；说明 shape 推断可能已足够准确，但也意味着方法在极端 shape 异常时的修正能力未验证。
- **依赖高质量伪 GT 训练**：扩散先验质量受训练用伪标签影响（SPIN/EFT），若观测域与训练域差距大，可能泛化受限。
- **未处理复杂相机内参/深度**：单目 setting 下相机参数简化为弱透视，真实场景（未知 intrinsics、多目标定缺失）需扩展。
- **未来方向**：(i) 扩展到 shape $\beta$ 联合建模；(ii) 引入更强的相机估计分支；(iii) 扩展到全身（手、脸）或细粒度关节；(iv) 探索与生成式 motion prior 的结合。

## 研究启发与可借鉴点
1. **"DDIM inversion → guided sampling loop"范式**：可用于其他参数化形体/几何恢复（如手、人脸、动物 mesh），只需更换 forward operator $\mathcal{A}$ 与 guidance loss。
2. **任务无关扩散先验 + 可插拔 guidance**：这种"一次训练、多处适配"的思路对多任务 pipeline 很有吸引力，可作为通用 refine 模块集成到现有 HMR 管线。
3. **用 $\nabla\|\mathbf{y}-\mathcal{A}(\hat{\mathbf{x}}_0)\|^2$ 近似后验 score**：该近似在观测噪声高斯假设下成立，且 $\hat{\mathbf{x}}_0$ 本身是图像条件的合理估计，这一技巧可迁移至其他 implicit prior-based inverse problem。
4. **交叉视角 / 时序一致性仅通过 $\mathcal{L}_{MV}/\mathcal{L}_{temp}$ 体现**：无需修改模型结构即可引入多帧或多视图约束，实现轻量级的"跨帧知识蒸馏"效果。

## 关键术语表
- **ScoreHMR**：本文提出的 Score-Guided Human Mesh Recovery，以扩散 score guidance 做 3D 人体参数迭代精化。
- **SMPL**：Skinned Multi-Person Linear model，参数化人体网格模型，包含 pose $\theta$ 与 shape $\beta$ 参数。
- **DDIM inversion**：确定性 DDIM 采样逆过程，可将干净样本映射到某一噪声层 latent。
- **Score guidance**：在扩散采样中叠加 $\nabla_{\mathbf{x}_t}\log p(\mathbf{y}|\mathbf{x}_t)$ 项，使采样结果与观测对齐。
- **PA-MPJPE**：Procrustes-aligned MPJPE，消除全局刚体变换后的关节误差（mm）。
- **MPJPE**：Mean Per Joint Position Error，关节绝对位置误差（mm）。
- **Acc Err**：加速度误差（$mm/s^2$），衡量时序平滑度。
- **$\mathcal{L}_{repr}$ / $\mathcal{L}_{MV}$ / $\mathcal{L}_{temp}$**：分别表示关键点对齐、多视角一致、时序一致三项 guidance loss。

## 可复现要素
- **数据集**：训练 Human3.6M、MPI-INF-3DHP、COCO、MPII（公开）；评测 3DPW、EMDB 1、Human3.6M、Mannequin Challenge（公开）。
- **代码/权重**：论文公开于 https://statho.github.io/ScoreHMR。
- **关键超参**：扩散时间步 $T$、引导步长 $\rho$、阈值 $\lambda_{thr}$、伪 GT 来源（SPIN/EFT）、图像 backbone 来源（ProHMR/PARE/HMR 2.0）；详细超参见 supplemental，正文未逐一列出。
