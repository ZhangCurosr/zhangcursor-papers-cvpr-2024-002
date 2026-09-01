---
title: "On-Exact-Inversion-of-DPM-Solvers"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Hong_On_Exact_Inversion_of_DPM-Solvers_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 15:19:21"
field: "扩散模型可逆性与图像编辑"
keywords: ["DPM-Solver", "diffusion inversion", "backward Euler", "classifier-free guidance", "tree-ring watermark", "latent diffusion model", "image editing"]
innovations: ["提出基于后向Euler+梯度下降的DDIM精确反演算法，支持大classifier-free guidance", "提出高阶DPM-Solver的高阶项近似反演策略，首次实现DPM-Solver++(2M)的精确反演", "提出VAE解码器的梯度下降伪逆，降低LDM反演重建误差下界"]
benchmarks: ["ImageNet64", "LDM (Stable Diffusion v1.4)", "Tree-Ring Watermark classification"]
---

# 论文速读：On Exact Inversion of DPM-Solvers

## 一句话总结
本文针对扩散概率模型（DPM）快速采样器（DPM-Solvers）难以精确反演的问题，提出了基于隐式方法（后向 Euler）的精确反演算法：对一阶 DDIM 使用后向 Euler+梯度下降/前向步法，对高阶 DPM-Solver++ 提出高阶项近似策略；在图像/噪声重建、Tree-ring 水印检测与分类、保背景图像编辑等任务上显著优于 Naïve DDIM 反演和前人方法。

## 研究问题与动机
- **DPM-Solver 加速导致不可逆性**：DDIM/DPM-Solver++ 等将去噪过程建模为 ODE 并用显式/多步求解器快速采样（10–50 步），但反向追溯初始噪声时，Naïve DDIM 反演依赖"相邻步噪声近似相等"的假设，步数越少误差越大。
- **已有精确反演方法的局限**：[34,40] 设计全新可逆采样流程，不兼容标准 DDIM/DPM-Solver 生成图像；[21] 使用不动点迭代（FPI）对 DDIM 可精确反演，但 classifier-free guidance ω>1 时严重发散/误差激增。
- **缺乏通用的高阶 DPM-Solver 反演**：高阶线性多步法（如 DPM-Solver++(2M)）依赖历史状态，反向时未来状态未知，现有工作无法处理。
- **应用驱动需求**：精确反演是图像编辑、风格迁移、水印溯源、模型攻击等应用的核心组件；尤其在 LDM+大 ω 场景下，亟需鲁棒的精确反演方法。

## 核心贡献（创新点）
1. **提出 DDIM 的后向 Euler 精确反演算法（Alg. 1）**：在每个反演步用梯度下降或前向步法求解隐式方程，本质区别在于用数值优化替代 Naïve DDIM 的前向 Euler 近似，且可稳定支持 ω>1 的大 guidance。
2. **提出高阶 DPM-Solver 的高阶项近似反演算法（Alg. 2）**：用细粒度 Naïve DDIM 反演估计未知的过去状态，再以后向 Euler 结合高阶项解析近似处理线性多步法的回溯，首次实现 DPM-Solver++(2M) 的精确反演。
3. **提出精确的解码器反演（$\mathcal{D}^{\dagger}$）**：对 LDM 的 VAE 解码器使用梯度下降求伪逆，替代前人直接使用编码器 E 的做法，降低因 autoencoder 信息损失带来的重建误差下界。
4. **系统验证三大应用**：图像/噪声重建 NMSE 显著降低；Tree-ring 水印实现分类（首次）；保背景图像编辑在 ω=7.5 下 NMSE 接近 Oracle 基线。

## 方法详解

### 4.1 DDIM 精确反演（Alg. 1）
- **问题形式化**：给定 $x_{t_i}$，求 $z_{t_{i-1}}$ 使得经正向一步后恰好得到 $x_{t_i}$。由 DDIM 采样公式（Eq.4）可得隐式方程：
  $$
  z'_{t_i} = \frac{\sigma_{t_i}}{\sigma_{t_{i-1}}}\hat{z}_{t_{i-1}} - \alpha_{t_i}(e^{-h_i}-1)z_\theta(\hat{z}_{t_{i-1}}, t_{i-1})
  $$
  目标是令 $z'_{t_i} = \hat{z}_{t_i}$。
- **求解策略**：先以 Naïve DDIM 反演（Eq.6）初始化 $\hat{z}_{t_{i-1}}$，再用梯度下降最小化 $\|\hat{z}_{t_i} - z'_{t_i}\|_2^2$，或用前向步法：
  $$
  \hat{z}_{t_{i-1}} \leftarrow \hat{z}_{t_{i-1}} - \rho(z'_{t_i} - \hat{z}_{t_i})
  $$
  直至收敛。
- **为何优于 FPI**：FPI 算子 $F$ 的非扩张性要求 $z_\theta$ 满足强 Lipschitz 条件；ω>1 时 classifier-free guidance 使 $z_\theta$ 的有效 Lipschitz 常数增大，FPI 难以收敛。梯度下降可通过减小步长保证收敛，鲁棒性更强。
- **解码器反演 $\mathcal{D}^{\dagger}$**：对 LDM，先由编码器得初始 latent，再梯度下降最小化 $\|x_0 - \mathcal{D}(z)\|_2^2$，比直接使用 $\mathcal{E}$ 误差更小。

### 4.2 高阶 DPM-Solver 精确反演（Alg. 2）
- **核心难点**：DPM-Solver++(2M) 的采样公式（Eq.5）使用 $x_{t_{i-1}}$ 和 $x_{t_{i-2}}$ 两步历史，反向时两者均未知。
- **解决思路**（图 1）：先用细粒度 Naïve DDIM 反演（更多步数）得到近似值 $\hat{y}_{t_{i-1}}, \hat{y}_{t_{i-2}}$ 替代未知的真实历史状态；然后在后向 Euler 框架中将高阶项视为常数进行近似（Eq.9）：
  $$
  d'_i \leftarrow z_\theta(\hat{z}_{t_{i-1}}, t_{i-1}) + \frac{z_\theta(\hat{y}_{t_{i-1}}, t_{i-1}) - z_\theta(\hat{y}_{t_{i-2}}, t_{i-2})}{2r_i}
  $$
  其中 $r_i = \frac{\lambda_{t_{i-1}} - \lambda_{t_{i-2}}}{\lambda_{t_i} - \lambda_{t_{i-1}}}$，再迭代更新 $\hat{z}_{t_{i-1}}$ 直至收敛。
- **与 Naïve 细化的本质区别**：单纯增加 Naïve DDIM 步数仅能逼近扩散 ODE 轨迹，无法修正 DPM-Solver 自身的离散误差；后向 Euler+高阶项近似则直接修正求解器误差。

## 实验与结果

### 数据集与设置
- **Pixel-space DPM**：ImageNet64 训练的一阶扩散模型 [31]。
- **LDM**：Stable Diffusion v1.4 [26]，classifier-free guidance ω=3.0（重建）和 ω=7.5（编辑）。
- **生成采样器**：DDIM 50 步、DPM-Solver++(2M) 10 步。
- **评估指标**：NMSF（噪声/图像重建 NMSE）、LPIPS、SSIM；水印用 $l_1$ 距离分类；保背景编辑用背景区域 NMSE。

### 主要结果
- **重建误差**（Fig.3）：Alg. 1（DDIM 50 步）和 Alg. 2（DPM-Solver++ 10 步）在像素空间和潜空间均显著低于 Naïve DDIM 反演；Naïve 步数增至 1000 后误差饱和（黑线），无法进一步降低。Pan et al. [21] 的 FPI 在 LDM+ω=3 下噪声重建性能很差（Fig.3c）。
- **水印检测与分类**（Fig.4,5）：Alg. 2 在 Tree-ring 水印重建质量上最优； confusion matrix 显示 Naïve DDIM 难以检测 WM 2，Alg. 2 可正确分类三种水印（首次实现水印分类）。
- **保背景图像编辑**（Table 3，60×5=300 张，ω=7.5）：

  | 方法 | 背景 NMSE (×10⁻³) |
  |---|---|
  | Oracle [23] | 11.0 ± 1.7 |
  | Naïve DDIM | 30.4 ± 4.8 |
  | Naïve + $\mathcal{D}^{\dagger}$ | 18.4 ± 2.0 |
  | **Alg. 1 (ours)** | **12.8 ± 2.0** |

  Alg. 1 在无需原始 latent 的情况下，背景保真度已接近 Oracle 水平。

- **速度**（Table 4，LDM）：Naïve DDIM（50 步）3s，FPI [21] 32s，Alg. 1 79s，Alg. 2 159s——计算代价显著增加，为已知局限。

## 相关工作脉络
1. **Naïve DDIM 反演**（广泛使用于 Prompt-to-Prompt 等编辑工作）：基于前向 Euler 的显式近似，步数少时误差大，本文 Alg. 1 以其为初始化但用隐式迭代大幅改进。
2. **EDICT [34] / 双向近似 [40]**：提出新采样流程实现可逆性，但仅适用于自身生成的图像，无法处理标准 DDIM/DPM-Solver 输出；本文方法通用性更强。
3. **AIDI [21]（Pan et al.）**：首个对标准 DDIM 图像的精确反演方法，使用 FPI；本文指出其在 ω>1 时因 Lipschitz 条件破坏而失败，改用梯度下降/前向步法克服。
4. **DPM-Solver / DPM-Solver++ [14,15]**：快速 ODE 采样器（10–50 步），本文首次给出其精确反演方案（Alg. 2）。
5. **Tree-ring Watermark [36]**：在初始噪声傅里叶空间嵌入水印；本文证明 Alg. 2 可支持水印分类（不仅是检测）。
6. **保背景编辑 [23]**：原始方法依赖 Oracle 轨迹；本文证明无需轨迹即可达到近似 Oracle 的保背景效果。

## 局限性与未来方向
- **计算成本高**：Alg. 1/2 耗时分别为 Naïve 的 16× 和 32×（Table 4），限制实时应用。
- **LDM 中非完美精确**：受限于 autoencoder 信息损失和加速调度器离散误差，实际为"近精确"（near-exact）反演，NMSE 难低于 10⁻⁶。
- **需预知 prompt**：LDM 反演假设已知生成时使用的文本 prompt。
- **联合估计未解决**：prompt 与初始噪声的联合估计留作未来工作。
- **高步数加速调度器下精确性下降**（Fig.3b）。

## 研究启发与可借鉴点
1. **隐式方法替代显式近似的反演范式**：将数值 ODE 求解中的后向 Euler + 梯度下降引入扩散模型反演，可有效克服大 guidance 下的收敛问题；该思路可迁移至 SDE-based 模型或其他隐式生成模型的反演。
2. **高阶项近似策略**：用细粒度低阶反演估计未知历史状态以支撑高阶反向，这一"分层近似"思想可推广至其他多步求解器的可逆性问题。
3. **解码器精确伪逆**：用梯度下降替代 VAE 编码器的反演设计，降低了 LDM 重建误差下界，可成为后续 LDM 反演工作的标准组件。
4. **实验设计的层次感**：从噪声重建→图像重建→水印分类→保背景编辑，层层递进验证方法有效性，并为"精确反演 enables new tasks"提供有说服力的证据链。
5. **与团队方向结合机会**：若团队关注扩散模型编辑/水印/安全，可将 Alg. 2 集成至现有 DPM-Solver  pipeline，作为通用反演模块；也可探索将隐式反演扩展至视频/3D 扩散模型。

## 关键术语表
- **DPM-Solver / DPM-Solver++**：将扩散去噪过程建模为 ODE，用指数积分器/线性多步法实现 10–50 步快速采样的训练无关加速方法。
- **Classifier-free Guidance (ω)**：无需额外分类器，通过条件/无条件预测的加权组合增强生成质量的技巧，ω>1 时提升 fidelity 但增加反演难度。
- **Naïve DDIM Inversion**：将 DDIM 采样公式逐项反转得到的显式反演，本质是前向 Euler 在时间反向上的应用，依赖相邻步噪声近似相等的假设。
- **Backward Euler（后向 Euler）**：隐式 ODE 求解方法，每步需解方程；本文用于反演时在每步用梯度下降求解，保证 ω>1 时的收敛稳定性。
- **Fixed Point Iteration (FPI)**：通过 $z \leftarrow F(z)$ 迭代收敛到不动点的方法；本文指出其在 ω>1 时因 Lipschitz 条件破坏而失效。
- **Tree-Ring Watermark**：在初始噪声傅里叶空间嵌入树环状隐形水印，用于生成图像版权溯源的方法 [36]。
- **LDM（Latent Diffusion Model）**：在压缩潜空间（而非像素空间）执行扩散过程的模型，如 Stable Diffusion，需 VAE 编解码器支持。
- **Decoder Inversion ($\mathcal{D}^{\dagger}$)**：通过梯度下降求解 $\min_z \|x - \mathcal{D}(z)\|_2^2$ 得到潜变量近似伪逆，优于直接使用编码器。

## 可复现要素
- **数据集**：ImageNet64（pixel-space DPM 训练）；LDM 使用 Stable Diffusion v1.4 官方权重；水印实验 3 种 watermark pattern（论文未公开具体数据，需按 [36] 复现）。
- **代码/权重**：项目页面 https://smhongok.github.io/inv-dpm.html（论文未明确说明 GitHub 仓库是否开源，以项目页面为准）。
- **关键超参**：DDIM 反演步数 50；DPM-Solver++(2M) 反演步数 10；fine-grained Naïve DDIM 步数未明确（Fig.3 黑线对应 1000 步）；LDM ω=3.0（重建）/ ω=7.5（编辑）；前向步法步长 ρ 未明确给出，论文未提及具体值。
