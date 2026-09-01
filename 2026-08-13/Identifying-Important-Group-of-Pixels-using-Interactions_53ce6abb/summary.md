---
title: "Identifying-Important-Group-of-Pixels-using-Interactions"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Sumiyasu_Identifying_Important_Group_of_Pixels_using_Interactions_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 12:25:26"
field: "视觉模型可解释性"
keywords: ["模型可解释性", "Shapley值", "交互作用", "Vision Transformer", "像素可视化", "博弈论"]
innovations: ["提出MoXI方法，利用博弈论交互作用识别对分类置信度有显著影响的像素组", "定义self-context/full-context变体将计算复杂度从指数级降至二次方级", "从博弈论角度证明贪心策略有效性并提供理论保证"]
benchmarks: ["ImageNet", "ViT-T", "DeiT-T", "ResNet-18"]
---

# 论文速读：Identifying-Important-Group-of-Pixels-using-Interactions

## 一句话总结
提出 MoXI（Model eXplanation by Interactions）方法，利用博弈论中的 Shapley 值和交互作用（interactions），高效识别对图像分类器预测置信度有显著影响的像素组；通过定义自上下文/全上下文变体将计算复杂度从指数级降至二次方级，在插入/删除曲线等评估中显著优于 Grad-CAM、Attention rollout 和 Shapley 值等基线。

## 研究问题与动机
1. 现有可视化方法（Grad-CAM、Attention rollout、Shapley values）仅基于单个像素的独立贡献排序，忽略了像素间的协同贡献与信息重叠，导致选出的"高贡献像素组"未必是最优信息集合。
2. 原始 Shapley 值和交互作用的计算需 O(|N|2^|N|) 次前向传播，难以直接用于高维图像的可解释性分析。
3. 仅选取 Shapley 值最高的像素集合会因信息冗余而效率低下——例如图1显示类别物体与背景存在显著协同效应，单独 ranking 会遗漏背景这一重要上下文。
4. 需要在保证理论合理性的前提下，设计高效、能捕捉像素集体贡献的可视化方法。

## 核心贡献（创新点）
1. **提出 MoXI 框架**：首次将博弈论交互作用引入像素组级别的可视化，同时考虑个体贡献与协同贡献，选出对分类置信度影响最大的像素集合；区别于 SHAP/Grad-CAM 仅 ranking 单像素。
2. **理论驱动的复杂度突破**：从博弈论角度证明简单贪心策略的最优性，自然导出 self-context（插入）和 full-context（删除）变体，将计算复杂度从指数级 O(|N|2^|N|) 降至二次方级 O(|N|²)；区别于 Prior 工作依赖重采样近似的路线。
3. **多维度实验验证**：在 ImageNet 上通过插入/删除曲线、Heatmap 可视化、常见腐蚀鲁棒性、跨类别数一致可解释性等实验，全面证明 MoXI 在识别重要像素组上的优势；最强结果（4% 可见像素达 90% 准确率）较第二强基线 Shapley values（25%）提升 65pp。

## 方法详解
方法名为 MoXI（Model eXplanation by Interactions），核心利用奖励函数 f(x) = log[P(y|x)/(1-P(y|x))]（目标类别对数几率），通过两种策略识别重要像素组：

**像素插入（Pixel Insertion）—— 贪心选增：**
- 问题：从全遮挡图像出发，逐步添加像素使 f(S) 最大增长。
- 第 k 步贪心选择：b_k = argmax_{b ∈ N\S_{k-1}} [φ^(0)(b) + I^(0)(S_{k-1}, b)]。
- 自上下文 Shapley 值：φ^(0)(a) = f(a) - f(∅)，衡量单像素独立贡献。
- 自上下文交互作用：I^(0)(a₁,a₂) = f(a₁∪a₂) - f(a₁) - f(a₂) + f(∅)，衡量两像素协同效应（正值=互补，负值=冗余）。
- 关键洞察：即使某像素 φ^(0) 很大，若与已选集合 S_{k-1} 信息重叠（I^(0) 为负），其联合增益可能很低，交互项能有效排除冗余。
- 复杂度：O(|N|²) 次前向传播。

**像素删除（Pixel Deletion）—— 贪心选删：**
- 问题：从完整图像出发，逐步移除像素使 f(N\S) 最小下降。
- 定义基于删除的 Shapley 值：φ_d(i|N) = Σ_{S⊆N,i∈S} P_d(S\{i}|N)[f(S) - f(S\{i})]，衡量移除像素的平均影响。
- 全上下文 Shapley 值：φ_d^(|N|)(a) = (1/|N|)[f(N) - f(N\{a})]。
- 全上下文交互作用：I_d^(|N|-1)(a₁,a₂) = (1/(|N|-1))[f(N) - f(N\{a₁}) - f(N\{a₂}) + f(N\{a₁,a₂})]。
- 贪心选择：b_k = argmax [φ_d^(|N|)(b) + (|N|-|S_{k-1}|)I_d^(|N|-|S_{k-1}|)(S_{k-1}, b)]，交互项带权重。
- 复杂度：同样 O(|N|²)。

**Masking 策略**：对 ViT 采用 feature patch masking（直接对 attention 施加 mask），避免 CNN 中 base value 替换引入的虚假特征；这是准确计算 Shapley 值与交互作用的前提。

## 实验与结果
**数据集与模型**：ImageNet 测试集（每类1张正确分类图像，共1000张）；主实验模型 ViT-T，附加 DeiT-T、ResNet-18。
**基线方法**：Grad-CAM、Grad-CAM++、Attention rollout、Shapley values、MoXI(-)（仅 self-context Shapley，无交互）。
**评估设置**：14×14 patches（16×16 像素/patch）；Shapley 值采样数 200；common corruptions level 5（fog、Gaussian noise 等）。

**主要结果**：
- **插入曲线**：仅 4% 可见像素时，MoXI 准确率 **90%**，Grad-CAM 2%、Attention rollout 4%、Shapley values 25%；MoXI 领先第二强基线 65pp。
- **删除曲线**：遮挡 10% 像素时，MoXI 准确率降至 **16%**，Grad-CAM/Attention rollout 仅降至约 79%。
- **可视化**：插入 Heatmap 中 MoXI 同时选中类别物体与背景；删除 Heatmap 中 MoXI 更集中于判别性区域。
- **一致性**：随训练类别数从10增至1000，MoXI 插入曲线准确率几乎不变，Attention rollout 显著下降。
- **鲁棒性**：fog/Gaussian noise 腐蚀实验中，MoXI 选中的像素被腐蚀后准确率下降最快，印证其识别的是真正关键像素。

## 相关工作脉络
1. **Grad-CAM / Grad-CAM++**：基于卷积层特征图梯度的可视化方法，权重由目标类别梯度决定；局限在于梯度幅度与真实贡献不对齐，且仅作用于 CNN。MoXI 直接基于 logits 计算，适用 ViT 并通过交互作用解决信息重叠问题。
2. **Attention rollout**：用于 ViT 的注意力流分析方法，累计 attention weight 作为贡献指标；与 MoXI 的本质差异在于后者基于博弈论给出理论保证的 pixel-level contribution，而非注意力权重的代理。
3. **SHAP / FastSHAP**：将 Shapley 值引入模型解释，但仅考虑单像素独立贡献且需大量采样近似；MoXI 引入交互作用并定义自上下文变体，无需重采样即可高效计算。
4. **RISE**：通过随机掩码样本 empirically 估计像素贡献；MoXI 从博弈论框架出发给出确定性贪心策略，具有更强的理论支撑。
5. **Zhang et al. (2021, ICLR)**：揭示交互作用与 dropout regularization 的相似性；本文进一步将交互作用用于像素组级别的可视化解释，而非模型正则化分析。
6. **Sumiyasu et al. (2022)**：利用交互作用分析模型误分类模式；本文将此视角推广至高效像素组识别，并解决计算复杂度瓶颈。

## 局限性与未来方向
**局限性**：
- Shapley 值通过随机采样近似（200次），非精确计算，可能引入估计误差。
- 实验主要聚焦 ViT 架构，对 CNN 的适配性虽用 ResNet-18 验证但未深入讨论 feature patch masking 在 CNN 中的可行性。
- 未充分讨论高分辨率图像下 O(|N|²) 复杂度的实际可扩展性（|N| 随分辨率平方增长）。
- 附录 G.2 提及对抗扰动的结果与常见腐蚀不同，但未深入分析交互作用在对抗场景下的行为。

**未来方向**：
- 探索更精确高效的 Shapley 值与交互作用计算方法（如多项式时间算法）。
- 将方法扩展至检测、分割等其他视觉任务。
- 结合人类主观评价，系统评估 MoXI 生成的 Heatmap 的可理解性。
- 研究交互作用在模型鲁棒性、对抗防御中的定量应用。

## 研究启发与可借鉴点
1. **理论驱动的高效近似设计**：从博弈论性质出发证明贪心策略的最优性，再导出 self-context/full-context 变体实现复杂度突破——这一"理论分析→结构简化→高效实现"的研究范式可迁移至其他需要近似博弈论量的任务（如特征选择、数据集压缩）。
2. **交互作用建模信息冗余**：通过 I(a₁,a₂) 显式捕获像素间的信息重叠，解决"高个体贡献≠高联合贡献"的痛点；该思路可直接应用于多模态融合、多传感器选择等需权衡互补/冗余信息的场景。
3. **多维评估体系**：插入/删除曲线 + Heatmap 可视化 + 腐蚀鲁棒性 + 跨配置一致性，构建了一套完整的可解释性评估框架；后续研究可借鉴此多角度验证策略，避免单一指标的局限性。
4. **Architecture-aware Masking 策略**：对 ViT 采用 feature patch masking 而非 base value 填充，避免了虚假特征的注入；这一实践提示不同架构需匹配相应的 masking 策略，可推广至 Mamba、ConvNeXt 等新架构的可解释性研究。
5. **Self-context 变体的通用价值**：自上下文 Shapley 值 φ^(0)(a)=f(a)-f(∅) 本质是"零背景参考"下的边际贡献，计算极轻量；这一简化思路可用于在线解释、实时可视化等对延迟敏感的应用场景。

## 关键术语表
**Shapley values（沙普利值）**：博弈论中衡量单个参与者对合作收益平均贡献的指标；本文用于量化每个像素对分类置信度的独立贡献。
**Interactions（交互作用）**：衡量两个或多个参与者协同贡献的博弈论量；本文用于捕捉像素间的合作效应与信息重叠（正值=互补，负值=冗余）。
**Self-context variants（自上下文变体）**：针对像素插入场景定义的 Shapley 值与交互作用简化形式（以自身为全集），计算复杂度从指数级降至二次方级。
**Full-context variants（全上下文变体）**：针对像素删除场景定义的 Shapley 值与交互作用形式，用于逐步移除像素的贪心策略。
**Insertion curve（插入曲线）**：从全遮挡图像开始，按方法排序逐步添加像素，记录分类准确率变化曲线；用于评估方法识别信息丰富像素的能力。
**Deletion curve（删除曲线）**：从完整图像开始，按方法排序逐步删除像素，记录分类准确率变化曲线；用于评估方法识别关键像素的能力。
**Feature patch masking（特征 patch 遮罩）**：对 ViT 的 attention 机制直接施加 mask，替代 CNN 中的 base value 填充，避免引入虚假特征。
**Consistent explainability（一致可解释性）**：模型在不同训练配置（如类别数量）下，方法选择的重要像素保持一致性；反映解释结果的稳定性与泛化性。

## 可复现要素
- **数据集**：ImageNet；测试子集为每类1张正确分类图像共1000张。
- **代码**：已开源，见 https://github.com/KosukeSumiyasu/MoXI。
- **权重**：使用标准 pretrained ViT-T、DeiT-T、ResNet-18 权重。
- **关键超参**：Patch 大小 16×16，网格 14×14；Shapley 值采样数 200；common corruptions level 5；类别数实验 10/20/100/1000。
- **实现细节**：feature patch masking 用于 Shapley 值与交互作用计算；近似 Shapley 值通过随机采样子集 S 计算。
