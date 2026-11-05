---
title: "RobustSAM-Segment-Anything-Robustly-on-Degraded-Images"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_RobustSAM_Segment_Anything_Robustly_on_Degraded_Images_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 08:34:22"
field: "鲁棒视觉分割"
keywords: ["鲁棒分割", "Segment Anything Model", "图像退化", "零样本分割", "特征一致性蒸馏", "傅里叶退化抑制"]
innovations: ["冻结SAM主干+轻量抗退化模块实现退化鲁棒零样本分割", "干净-退化双路径特征一致性损失蒸馏机制", "Fourier频域退化抑制模块用于分割特征去噪"]
benchmarks: ["Robust-Seg", "MSRA10K", "LVIS", "NDD20", "STREETS", "FSS-1000", "COCO", "BDD-100k", "LIS"]
---

# 论文速读：RobustSAM: Segment Anything Robustly on Degraded Images

## 一句话总结
本文提出 **RobustSAM**，在冻结预训练 SAM 主干的前提下，仅新增轻量模块（403 MB）并通过一致性损失训练，显著提升 SAM 在多种合成与真实退化图像上的零样本分割鲁棒性，同时构建 688K 规模的 **Robust-Seg** 数据集作为新基准。

## 研究问题与动机
- SAM 在低光照、噪声、模糊、恶劣天气、压缩伪影等退化条件下性能显著下降（引用 [24, 58, 61, 71]），直接影响下游任务（去雾、去模糊、超分）所依赖的结构先验。
- 先用图像恢复网络预处理再喂入 SAM 的策略不可靠：现有恢复算法以人眼感知为目标，不保证提升 SAM 分割性能（[7, 12, 40, 64, 70]）。
- 直接微调 SAM 全量参数或 Decoder 会引发灾难性遗忘，破坏其零样本泛化能力；而盲目地在退化数据上微调易导致已知知识丢失（[16, 30]）。
- 现有鲁棒分割方法（QualNet、URIE、FIFO）多聚焦单一退化类型，且联合下游任务训练会稀释 SAM 的零样本优势，缺乏对多类型退化的统一处理。

## 核心贡献（创新点）
1. **提出 RobustSAM 框架**：在冻结原始 SAM 主模块的基础上，仅引入 Anti-Degradation Token Generation（AOTG）、Anti-Degradation Mask Feature Generation（AMFG）及可微调的 Robust Output Token（ROT）三个轻量模块，以极低参数增量实现退化鲁棒分割。与直接微调 SAM 全量参数的本质区别在于：冻结主干保障了零样本能力的完整保留，仅训练新增模块。
2. **设计双重一致性蒸馏机制**：通过 Mask Feature Consistency Loss（$ \mathcal{L}_{\mathrm{MFC}} $）和 Token Consistency Loss（$ \mathcal{L}_{\mathrm{TC}} $）迫使退化路径提取的特征与 SAM 在干净图像上的特征对齐，本质区别于传统对抗/域适应方法——不依赖额外判别器，而是以干净 SAM 特征为教师信号做特征级一致性约束。
3. **提出 Fourier Degradation Suppression 模块**：将退化视为"图像风格"，在频域利用幅度分量隔离/滤除退化信息、保留相位结构，再通过逆变换回空间域，这是将频域退化抑制融入分割特征提取的首次系统性尝试。
4. **构建 Robust-Seg 数据集**：整合 7 个数据集共 43K 标注图像，施加 15 类合成退化，生成 688K 图像-掩码对，填补了多退化鲁棒分割基准的空白。
5. **证明对下游任务的正向增益**：以 RobustSAM 输出作为结构先验，在单图像去雾（SOTS）和去模糊（GoPro）任务上分别将 PSNR 提升 1.48 dB / 1.86 dB，表明鲁棒分割可惠及更广的退化图像应用链条。

## 方法详解
- **整体架构**：输入退化图像 → 冻结的 SAM Image Encoder（ViT）提取特征 → 新增的 ROT 替代原始 Output Token，经 Prompt Encoder 与 decoder 得到退化路径的 Mask Feature $ F_{MFD} $ 与 Token $ T_{RO} $；同时并行地，原 SAM 处理干净图像提取 $ F_{MFC} $、$ F_{CFC} $、$ T_{OC} $ 作为监督目标。
- **AMFG 模块**（Anti-Degradation Mask Feature Generation）：
  - 并行走 **Instance Normalization（IN）**（消除退化风格，保留内容）和 **Batch Normalization（BN）**（防止 IN 丢失细节）两条分支。
  - 通过注意力机制加权融合两者，再与原始输入沿通道拼接。
  - 接入 **Squeeze-and-Excitation Channel Attention（SEC）** 自适应调整通道权重。
  - 串联 **Fourier Degradation Suppression**：FFT 变换 → 1×1 卷积作用于幅度谱滤除退化风格 → 保留相位 → IFFT 回空间域。
  - 分别作用于互补特征 $ F_{CFD} $ 和掩码特征 $ F_{MFD} $，输出 $ \hat{F}_{CFD} $、$ \hat{F}_{MFD} $。
- **AOTG 模块**（Anti-Degradation Output Token Generation）：多层 IN + 单层 MLP，轻量过滤退化敏感信息，输出 $ \hat{T}_{RO} $。
- **Feature Fusion**：参照 HQ-SAM [27]，将 $ \hat{F}_{MFD} $、$ \hat{F}_{CFD} $ 与 $ \hat{T}_{RO} $ 融合为最终鲁棒掩码特征，生成预测 mask。
- **损失函数**：
  $$ \mathcal{L}_{\mathrm{Overall}} = \mathcal{L}_{\mathrm{MFC}} + \lambda_1 \mathcal{L}_{\mathrm{TC}} + \lambda_2 \mathcal{L}_{\mathrm{Seg}} $$
  $$ \mathcal{L}_{\mathrm{MFC}} = \|\hat{F}_{CFD} - F_{CFC}\|_2 + \|\hat{F}_{MFD} - F_{MFC}\|_2 $$
  $$ \mathcal{L}_{\mathrm{TC}} = \|\hat{T}_{RO} - T_{OC}\|_2 $$
  $$ \mathcal{L}_{\mathrm{Seg}} = \mathcal{L}_{\mathrm{Dice}}(P, G) + \lambda_3 \mathcal{L}_{\mathrm{Focal}}(P, G) $$
- **训练细节**：SAM 主干完全冻结；学习率 5e-4，40 epoch / 130K iter；8×A100，30 小时完成；退化增强覆盖 15 类合成退化 + 恒等映射（保证干净场景不降级）。推理时仅使用 RobustSAM（上半部分）。

## 实验与结果
- **数据集**：训练用 MSRA10K、ThinObject-5k、LVIS 的完整训练集；验证用 MSRA10K 和 LVIS 测试集；零样本测试用 NDD20、STREETS、FSS-1000、COCO（全部含合成退化）；真实退化测试用 BDD-100k、LIS。
- **基线**：SAM、HQ-SAM、AirNet+SAM（两阶段恢复+分割）、URIE+SAM。
- **MSRA10K（点提示，表2）**：RobustSAM **Degrade IoU=0.8609** / Average IoU=0.8616；对比 SAM Degrade 0.8194，**提升 +4.15 pp**；Clean 场景也优于 SAM（0.8726 vs 0.8402）。
- **LVIS（Box 提示，表3）**：RobustSAM Degrade IoU=0.7506，Average IoU=0.7511，均排名第一；SAM Degrade 仅 0.7341。
- **NDD20+STREETS+FSS-1000（点提示，表4）**：RobustSAM Degrade IoU=0.8195 / Average=0.8216；对比 SAM Degrade 0.7981，**提升 +2.14 pp**。
- **COCO（Box 提示，表5）**：RobustSAM AP=0.5130，对比 SAM AP=0.5002，**提升 +1.28 pp**；APL（大目标）达 0.5518 并列最优。
- **真实退化（BDD-100k+LIS，点提示，表6）**：RobustSAM **IoU=0.3717 / Dice=0.8926**，对比 SAM IoU=0.3056 / Dice=0.3837，**IoU 提升 +21.6%**，Dice 提升 **+132.7%**（Dice 基数低故增幅极大）；Box 提示下 IoU=0.8958 / Dice=0.9416。
- **下游任务（表8）**：去雾 PSNR 23.159 vs 21.677（**+1.48 dB**），SSIM 0.8685 vs 0.8451；去模糊 PSNR 29.351 vs 27.491（**+1.86 dB**），SSIM 0.9229 vs 0.9066。
- **消融（表7）**：微调 SAM 全量导致 IoU 骤降至 0.1871；仅微调 Output Token 为 0.3194；逐步加入 AMFG→AMFG+F→+AOTG→+ROT 各项均正向增益，验证各模块有效性。

## 相关工作脉络
1. **SAM [29]**：本文基础模型；定位差异——SAM 面向干净图像设计，鲁棒性不足；本文在其之上叠加退化抗性模块。
2. **HQ-SAM [27]**：高质量 SAM 变体，改进 mask decoder 结构；本文与其正交，可结合使用（未作消融对比）。
3. **AirNet [37]**：All-in-One 图像恢复；定位差异——以人眼视觉质量为优化目标，不直接面向分割下游，先恢复后分割的两阶段策略本文明确指出不可靠。
4. **URIE [64]**：面向视觉识别的通用图像增强；定位差异——URIE 针对分类任务优化，本文方法直接以分割特征一致性为监督信号。
5. **QualNet [28] / FIFO [35]**：分别通过可逆编码和雾特征滤波提升单类退化鲁棒性；本文覆盖 15 类退化，且保持 SAM 零样本能力不被稀释。
6. **MPRNet [87] / HINet [6]**：多类型图像恢复网络；定位差异——同样是感知导向恢复，本文不走恢复路线，而是学习退化不变特征。

## 局限性与未来方向
- 训练所用 15 类退化均为**合成退化**，与真实复杂退化（如运动模糊、镜头畸变、传感器噪声混合）之间仍存在 gap（论文声称泛化到 BDD-100k/LIS 实退化有提升，但未系统分析域偏置）。
- Fourier 频域滤波对非周期/结构性退化（如几何形变、压缩块效应）的抑制能力有限，未讨论。
- 仅测试了点提示和 Box 提示，**文本提示**下的鲁棒性未验证。
- 模型仅针对单一退化类型图像的鲁棒性提升，对**复合退化**（同时存在模糊+噪声+低光）的系统性评估较粗略。
- 688K 数据集虽大但来源有限（7 个数据集），未来可接入更多域（医学、遥感、水下）。
- 未来方向：扩展到视频分割、多模态提示（文本+点+框联合）、真实退化合成、与扩散先验结合。

## 研究启发与可借鉴点
1. **冻结主干+添加轻量一致性蒸馏模块**的策略可迁移至其他 foundation model 的退化鲁棒化（如 ViT 分类器、DINOv2），避免灾难性遗忘的同时获得显著提升。
2. **Fourier Degradation Suppression**思想（将退化视为频域风格，用幅度滤波+相位保留）可推广至其他视觉任务的去噪/去模糊前端设计。
3. **干净-退化双路径一致性损失**（教师-学生式特征对齐）是一种无需额外标注的自监督鲁棒化范式，可应用于医疗影像、卫星影像等高质量标注稀缺领域。
4. 将鲁棒分割作为**下游恢复任务的先验增强模块**的思路（本文第 5.3 节）可直接复用于去雨、去雪、低光照增强等任务，形成"鲁棒感知→更好先验→更好恢复"的正反馈闭环。
5. 训练效率对比极具参考价值：SAM 需 256 GPU 训练，RobustSAM 仅需 8 GPU 30 小时，这种"小增量高效适配"模式对资源受限团队非常友好，可作为后续轻量化工作的 baseline。

## 关键术语表
- **SAM（Segment Anything Model）**：Meta 提出的图像分割基础模型，基于 SA-1B 数据集预训练，支持点/框/文本提示的零样本分割。
- **RobustSAM**：本文提出的 SAM 鲁棒增强版本，通过轻量模块学习退化不变特征，保持零样本能力的同时提升对退化图像的分割性能。
- **AMFG（Anti-Degradation Mask Feature Generation）**：抗退化掩码特征生成模块，通过 IN+BN 并行、注意力融合、SEC 通道注意力与傅里叶退化抑制提取退化不变特征。
- **AOTG（Anti-Degradation Output Token Generation）**：抗退化输出 Token 生成模块，通过 IN+MLP 轻量过滤退化敏感信息。
- **ROT（Robust Output Token）**：可微调的输出 Token，替代原始 SAM 的固定 Output Token，使模型适应退化场景。
- **L_MFC（Mask Feature Consistency Loss）**：掩码特征一致性损失，约束退化路径输出特征与干净 SAM 特征在 L2 意义下对齐。
- **L_TC（Token Consistency Loss）**：Token 一致性损失，约束退化路径的 Token 与干净 SAM 的 Token 对齐。
- **Robust-Seg**：本文构建的 688K 图像-掩码对数据集，含 7 个源数据集及 15 类合成退化，面向退化鲁棒分割基准。

## 可复现要素
- **数据集**：Robust-Seg（688K）——论文声明构建但未明确说明是否公开；测试集使用 MSRA10K、LVIS、NDD20、STREETS、FSS-1000、COCO、BDD-100k、LIS，均为公开数据集。
- **代码**：论文未明确声明开源仓库链接（论文正文及附录未给出 GitHub 地址）。
- **权重**：论文未明确声明权重是否公开。
- **关键超参**：学习率 5e-4，40 epoch，130K iterations，batch size 8，8×A100，30 小时训练；退化增强 15 类 + 恒等映射；Loss 权重 λ₁–λ₃ 论文未给出具体数值（需在代码中确认）。
- **prompt 类型**：点提示（point prompt）和边界框提示（bounding box prompt）。
