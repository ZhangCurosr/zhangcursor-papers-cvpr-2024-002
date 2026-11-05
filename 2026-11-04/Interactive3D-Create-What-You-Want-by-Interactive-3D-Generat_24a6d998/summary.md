---
title: "Interactive3D-Create-What-You-Want-by-Interactive-3D-Generat"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Dong_Interactive3D_Create_What_You_Want_by_Interactive_3D_Generation_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 00:42:21"
field: "3D 生成与编辑"
keywords: ["Interactive 3D Generation", "Gaussian Splatting", "InstantNGP", "Score Distillation", "3D Editing", "Text-to-3D", "Hash Refinement"]
innovations: ["两阶段异构表示（Gaussian Splatting + InstantNGP）实现可交互且高质量的 3D 生成", "局部专属哈希与残差 MLP 解决多区域同时精炼时的特征冲突问题"]
benchmarks: ["CLIP R-Precision on Cap3D-derived 50 prompts"]
---

# 论文速读：Interactive3D: Create What You Want by Interactive 3D Generation

## 一句话总结
论文提出了 **Interactive3D** 框架，通过两阶段异构 3D 表示（Gaussian Splatting → InstantNGP）实现用户在生成过程中的直接、精细交互控制，支持添加/移除部件、拖拽变形、刚性移动、局部语义编辑等操作，显著提升了文本到 3D 生成的可控性与质量（CLIP R-Precision 达 0.94）。

---

## 研究问题与动机

1. **现有 3D 生成方法缺乏精确用户控制**：主流方法（DreamFusion、ProlificDreamer 等）仅依赖初始文本提示或 2D 参考图，无法在生成过程中灵活干预，导致结果与用户预期存在偏差。
2. **文本提示无法表达复杂 3D 设计**：纯文本条件在刻画精细几何结构与空间关系上能力有限，而 2D 参考图又无法覆盖完整三维结构，可能引入 undesired artifacts。
3. **3D 领域缺乏类似 Stable Diffusion 的基础模型**：直接移植 ControlNet 思路面临两大障碍——(i) 3D 控制信号采集成本远高于 2D；(ii) 3D 基础模型尚未成熟，fine-tuning 条件不具备。
4. **NeRF 隐式表示不友好于交互，Gaussian Splatting 又难以提取高质量 Mesh**：前者交互困难，后者几何表达与网格提取仍是未解问题，需要一种互补的两阶段方案。

---

## 核心贡献（创新点）

1. **两阶段异构表示框架**：首次系统性地将 Gaussian Splatting（面向交互）与 InstantNGP（面向几何细化与 Mesh 提取）串联为统一的交互式 3D 生成管线，两者优势互补。
2. **五大交互式原语**：提出可添加/移除部件、可变形拖拽（Deformable Dragging）、刚性拖拽（Rigid Dragging，带刚性约束损失）、几何变换与局部语义编辑五种交互操作，并在 Gaussian Splatting 表示下实现。
3. **Interactive Hash Refinement 模块**：针对 InstantNGP 哈希冲突与特征容量瓶颈，设计基于部分占用网格的局部多分辨率哈希表与残差 MLP，实现"冻结满意区域、仅精炼目标区域"的无退化局部优化。
4. **Interactive SDS Loss + 自适应相机近焦策略**：通过调整去噪步 t 与将相机移至修改区域附近进行局部渲染，显著提升局部交互的优化效率，整体生成时间降至约 50 分钟。
5. **SOTA 性能**：在 CLIP R-Precision 上达到 0.94，超越 ProlificDreamer（0.83）与 DreamFusion（0.67），同时保持较高生成效率。

---

## 方法详解

### 总体架构（两阶段级联）
- **Stage I（Gaussian Splatting 交互阶段）**：以 $N$ 个高斯球 $\mathcal{E} = \{(c_i, o_i, \mu_i, \Sigma_i)\}_{i=1}^N$ 表示 3D 对象，视其中心点构成点云 $\mathcal{S}=\{\mu_i\}$ 以支持交互操作；共优化 10k 步。
- **Stage II（InstantNGP 细化阶段）**：通过 NeRF Distillation 将 Gaussian 蒸馏为 InstantNGP 表示，再进行 Interactive Hash Refinement 细化，共优化 10k 步；总步数 20k，运行于 NVIDIA A100。

### Stage I 关键设计

**1) 添加/移除部件**
- 添加：简单拼接两组高斯集合 $\mathcal{E}_1 \cup \mathcal{E}_2$。
- 移除：按 2D 投影掩码（借助 SAM 多视图分割）或 3D 手选区域定义 $I(P)$，执行 $\mathcal{E}' = \mathcal{E} - \{e_i\}_{i \in I(P)}$（公式 1）。

**2) 几何变换**
对选定部件 $P$ 构建包围盒 $B(P)$，施加旋转/平移/拉伸变换 $\tau$ 后拼回：$\mathcal{E}_{\text{trans}} = \tau(B(P)) + \mathcal{E}'$（公式 2）。

**3) 可变形 / 刚性拖拽（Inspired by DragGAN）**
- 激活区域：$P = \{e_i \mid \|\mu_i - p_s\|_2 \leq r\}$（公式 3）。
- 每步沿 $p_s \to p_t$ 方向微移：$\mu_i' = \mu_i + \alpha \frac{(p_t - p_s)}{\|p_t - p_s\|_2}$（公式 4）。
- 运动监督损失：$L_{\text{motion}} = \sum_{i \in I(P)} \|\mu_i' - p_t\|_1$（公式 5）。
- 刚性约束损失（防形变）：$L_{\text{rigid}} = \sum_{i \in I(P)} \big|\|p_s - p_i\|_2 - \|p_s^* - p_i^*\|_2\big|$（公式 6），维持邻点间相对距离不变。
- 可变形模式持续更新 $p_s$ 以捕获新点；刚性模式固定 $p_s$ 并引入 $L_{\text{rigid}}$。拖拽后执行 densification/pruning 填补空洞。

**4) 局部语义编辑**
选定部件后输入新文本 prompt，通过 SDS loss 驱动局部视觉属性变化（如 "make wings aflame"）。

**5) Interactive SDS Loss**
- 自适应相机 Zoom-in：将相机移至修改区域附近，以局部渲染计算 SDS，提升效率。
- 可调节去噪步 $t$：早期粗生成取大 $t$（如 0.98），后期微调取小 $t$（如 0.3），避免剧烈扰动。

### Stage II 关键设计

**1) NeRF Distillation（高斯 → InstantNGP）**
以高斯渲染图监督 InstantNGP 随机视角渲染：$L_{\text{distill}} = \frac{1}{M}\sum_c \|\mathcal{V}(\mathcal{F}, c) - \mathcal{R}(\mathcal{E}, c)\|_1$（公式 7），实现快速几何初始化。

**2) Interactive Hash Refinement**
- 从 InstantNGP 提取 $32^3$ 二值占用网格 $O$；给定感兴趣区域 $Q=\{p \mid \|p-o\|_2 \leq r\}$（公式 8）。
- 取交集 $O^{\text{part}} = O \cap Q$，划分多分辨率网格，建立**局部专属**哈希表 $\mathcal{H}^{\text{part}}$ 与特征 $\mathcal{F}^{\text{part}}$（公式 9）：
  $$f_k = \begin{cases} H_k^{\text{part}}(p, \mathcal{F}_k^{\text{part}}), & p \in O^{\text{part}} \\ 0, & p \notin O^{\text{part}} \end{cases}$$
- 新增轻量 MLP 学习残差颜色与密度，叠加到原有 InstantNGP 输出，**避免全局哈希冲突扩散至未修改区域**。
- 随后以 Interactive SDS Loss 对局部区域进一步优化。

---

## 实验与结果

- **评估指标**：CLIP R-Precision（遵循 Shap-E [10] 评测协议），使用 Cap3D [18] 派生的 50 条 prompt。
- **基线对比**：
  | 方法 | R-Precision | 平均耗时 |
  |---|---|---|
  | DreamFusion [27] | 0.67 | 1.1 h |
  | ProlificDreamer [35] | 0.83 | 3.4 h |
  | **Ours** | **0.94** | **50 min** |
- **主要结论**：
  - Interactive3D 以 0.94 取得最高 CLIP R-Precision，较 ProlificDreamer 提升 **+0.11**，较 DreamFusion 提升 **+0.27**。
  - 生成效率最高（50 min < 1.1 h / 3.4 h），得益于：① Stage I 快速高斯优化；② Stage I 结果初始化 Stage II，加速收敛；③ 交互嵌入现有 checkpoint，无需额外训练步。
- **定性展示**：支持组合生成（骑士 + 火龙 dragon）、姿态调整（站立→踢舞）、刚性拖拽（霸王龙头部转向）、局部细化（Gundam 机器人纹理/几何提升）等场景。

---

## 相关工作脉络

1. **DreamFusion / ProlificDreamer / Fantasia3D / Magic3D**：基于 SDS/VSDS 优化的文本到 3D 方法，仅依赖初始文本，**无生成中交互能力**；本文在其基础上注入可中断式 3D 交互。
2. **PointE / Shap-E / One-2-3-45**：Feed-forward 3D 生成，速度快但可控性弱；本文属优化范式，强调**用户闭环反馈**。
3. **ControlNet [38]**：2D 图像控制的 fine-tuning 范式；本文指出 3D 控制信号采集成本高且 3D 基础模型欠缺，故转向**直接在生成流中集成交互原语**的策略。
4. **DragGAN [26]**：2D 图像的 point-based 拖拽编辑；本文将其 3D 化，利用 3D 点天然可追踪的优势免去 2D 跟踪流程。
5. **3D Gaussian Splatting [11]**：显式、可编辑的高斯表示；本文取其交互友好性，但补充 InstantNGP 阶段以克服其在几何/Mesh 提取上的不足。
6. **InstantNGP [23]**：多分辨率哈希编码加速 NeRF；本文利用其几何与网格提取优势，并通过局部专属哈希解决"多区域同时精炼时特征冲突"的问题。

---

## 局限性与未来方向

1. **Stage I 仍可能产生长尾噪声与伪影**：SDS 优化本身的不稳定性导致高斯球优化过程震荡，需要 Stage II 进一步去除（论文 Fig. 4(a) 可见模糊与 artifacts）。
2. **部件选择依赖 SAM 等预训练分割器**：2D mask-based 的部件选取需推理大模型，速度较慢；早期生成阶段渲染图信息不足时分割精度下降，3D 点选替代方案精度亦有限。
3. **InstantNGP 局部哈希容量仍需手工设定**：哈希表级数与 feature capacity 影响细化质量，尚依赖经验调参，缺乏自适应容量分配策略。
4. **当前交互集中在形状/纹理层面**：对拓扑结构变化、材质物理属性、动画绑定等更高维交互未涉及。
5. **可扩展性待验证**：论文主要展示单对象生成；多对象复杂场景、人物全身高保真生成等仍需更多验证。

---

## 研究启发与可借鉴点

1. **"交互友好表示 + 几何友好表示"的两阶段范式**：将 Gaussian Splatting 与 InstantNGP 串联，既保留显式 blob 的可编辑性，又获得隐式场的网格提取能力，这一**异质表示级联**思想可迁移至任何需要"中间可交互 + 最终高质量输出"的 3D 生成任务。
2. **Interactive SDS Loss 与 adaptive camera zoom-in**：在局部交互区域切换近焦相机 + 局部渲染计算 SDS，可显著减少无关节梯度的浪费；该策略可用于其他基于 diffuison prior 的 3D 优化任务（如 3D 编辑、inpainting）。
3. **Part-specific hash mapping 解决多区域同时优化的冲突问题**：局部专属哈希 + 残差 MLP 的设计，本质上是一种"区域隔离的增量式特征注入"，可直接迁移至 InstantNGP-based 的 3D 编辑、局部 inpainting、风格迁移等下游任务。
4. **Rigid vs. Deformable dragging 的统一损失设计**：通过 $L_{\text{motion}}$ 与 $L_{\text{rigid}}$ 的组合实现同一接口下的两种行为，这种**单一界面、多模式切换**的交互设计对 3D 编辑工具开发具有参考价值。
5. **可调去噪步 t 以适应不同生成阶段**：早期大 t 保多样性、后期小 t 保稳定性的思路，可推广到任意基于 Score Distillation 的渐进式 3D 生成流程中。

---

## 关键术语表

**Gaussian Splatting**：用一组可微分 3D 高斯椭球显式表示场景并通过 splatting 投影到 2D 图像的最新渲染技术，具有良好的可编辑性。
**InstantNGP**：基于多分辨率 hash 编码的 NeRF 加速版本，利用 hash 表将 3D 坐标映射到可学习特征，实现高效细粒度场景表示与网格提取。
**SDS Loss (Score Distillation Sampling)**：利用预训练 2D 扩散模型的梯度对 3D 参数化渲染器进行优化的核心损失函数，是 DreamFusion 系列方法的基础。
**Interactive Hash Refinement**：本文提出的局部哈希细化模块，通过在选定区域构造独立多分辨率哈希表与残差 MLP，实现无退化的局部细节增强。
**Deformable Dragging**：类 DragGAN 的 3D 拖拽方式，允许源点随优化动态更新，使局部结构发生塑性形变。
**Rigid Dragging**：刚性拖拽方式，通过刚性约束损失保持选中区域内点间相对距离不变，实现整体平移/旋转。
**CLIP R-Precision**：以 CLIP 相似度为检索指标的召回率评估，常用于文本到 3D 生成质量衡量（本文用作主要定量指标）。
**NeRF Distillation**：以 Gaussian Splatting 渲染图为监督信号训练 InstantNGP 的参数蒸馏过程，用于 Stage I 到 Stage II 的表示转换。

---

## 可复现要素

- **数据集**：评估使用基于 Cap3D 派生的 50 条 prompt；训练阶段未指定特定数据集（依赖预训练 2D diffusion prior）。
- **代码开源**：项目页面 https://interactive-3d.github.io/，论文未明确声明代码开源仓库地址。
- **权重开源**：论文未声明公开预训练权重。
- **关键超参**：
  - 总优化步数：20k（Stage I 10k + Stage II 10k）
  - 硬件：NVIDIA A100 GPU
  - Occupancy grid 分辨率：$32 \times 32 \times 32$
  - 交互去噪步 t：早期 ~0.98，微调 ~0.3
  - 拖拽步长超参 α：论文未给出具体数值（见 supplementary）
  - 部件半径 r、移动步数等：论文未列出具体数值

---
