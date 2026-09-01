---
title: "MFP-Making-Full-Use-of-Probability-Maps-for-Interactive-Imag"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Lee_MFP_Making_Full_Use_of_Probability_Maps_for_Interactive_Image_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 12:26:20"
field: "交互视觉分割"
keywords: ["交互图像分割", "概率图调制", "点击交互", "递归训练", "Late Fusion"]
innovations: ["提出概率图调制机制充分利用历史预测形状信息", "设计Late Fusion融合策略强化概率特征影响", "开发递归训练方案端到端学习完整交互流程"]
benchmarks: ["GrabCut", "Berkeley", "DAVIS", "SBD"]
---

# 论文速读：MFP-Making-Full-Use-of-Probability-Maps-for-Interactive-Imag

## 一句话总结
本文提出MFP（Making Full Use of Probability Maps）框架，通过概率图调制技术充分挖掘历史概率图形状信息，在相同骨干网络下显著优于现有点击交互分割方法，可在更少点击次数下达到更高IoU。

## 研究问题与动机
- **历史概率信息利用率低**：现有方法虽将上一轮概率图作为网络输入，但未有效提取其中的形状细节信息，导致当前预测无法充分利用历史预测结果。
- **点击信息与形状信息割裂**：用户点击提供精确的前景/背景区分信息，而概率图蕴含目标物体形状信息，二者缺乏有效融合机制。
- **训练策略不足**：早期工作采用随机采样+迭代错误点击的方式训练，初始点击无历史概率图可供调制，无法端到端学习完整交互流程。
- **公平对比困难**：骨干网络强度差异导致方法间性能比较失真，需在相同Backbone下验证算法有效性。

## 核心贡献（创新点）
1. **提出概率图调制（Probability Map Modulation）机制**：首次利用伽马校正对历史概率图进行调制，增强目标物体形状表征，与已有工作直接拼接概率图形成本质区别。
2. **设计Late Fusion融合策略**：将调制后概率特征与骨干特征在分割头前融合，强化概率信息对最终预测的影响，而非仅作为输入通道。
3. **开发递归训练（Recursive Training）方案**：完全基于有序点击序列进行训练，从目标中心开始按预测误差递归采样，充分利用每一轮概率信息。
4. **多骨干通用性验证**：在ResNet-34、HRNet-18、ViT-B三种主流骨干上实现MFP，证明框架的泛化能力。

## 方法详解
### 概率图调制（Probability Map Modulation）
- **调制窗口定义**：$M = \{x : \|x - u\| \leq R\}$，以当前点击$u$为中心、半径$R$为范围的区域。
- **前景点击调制**（$l(u) = 1$）：$\tilde{P}_x^{t-1} = (P_x^{t-1})^{\frac{1}{\gamma}}$，提升概率值趋向1。
- **背景点击调制**（$l(u) = 0$）：$\tilde{P}_x^{t-1} = (P_x^{t-1})^{\gamma}$，降低概率值趋向0。
- **伽马分配策略**：
  - **欧氏距离**（后期点击）：$\gamma = \Gamma \cdot (1 - \frac{d}{R}) + \frac{d}{R}$，线性衰减。
  - **概率距离**（前期点击）：$\gamma = (\Gamma - 1) \cdot \frac{(\bar{d} - d)^3}{\bar{d}^3} + 1$，基于概率差异的三次方衰减。
- **窗口半径自适应**：$R = \min\{\frac{1}{2} \cdot \min_{c \in O}\|u - c\|, R_{\max}\}$，避免与异类点击重叠。

### 网络架构
- **输入处理**：将图像$I$、点击图$C^t$、原概率图$P^{t-1}$、调制概率图$\tilde{P}^{t-1}$分别嵌入相同维度后逐元素相加输入骨干网络。
- **Late Fusion**：提取概率相关特征$\mathcal{F}_P^t$后与骨干特征$\mathcal{F}_B^t$拼接，经Xception conv blocks融合后输入分割头。

### 递归训练策略
1. 初始点击采样于目标物体中心。
2. 计算预测与Ground Truth误差，在最大误差区域中心选取下一点击。
3. 递归执行至最多24次点击，全程利用历史概率图进行调制。

## 实验与结果
### 数据集与训练设置
- **测试集**：GrabCut、Berkeley、DAVIS、SBD
- **训练集**：SBD（8,498图像）；COCO + LVIS（104K图像，1.6M实例掩码）
- **评估指标**：NoC@85/90/95（达到指定IoU所需平均点击数）、mIoU-AUC

### 主要结果
| 训练数据 | 最优NoC@90提升 | 备注 |
|---------|---------------|------|
| SBD | MFP(ViT-B): 5.32 vs SimpleClick: 5.48 | 12项中9项最佳，3项第二 |
| COCO+LVIS | MFP(ViT-B): 4.81 vs SimpleClick: 5.06 | 12项中7项最佳，5项第二 |

- **相同骨干公平对比**（Table 4）：36项评测中28项最优，残差项差距<0.1。
- **定性分析**：MFP能正确分割远离点击的区域（如自行车轮），而对比方法失败。

### 超参数
- $N = 7$（概率距离切换至欧氏距离的点击轮次）
- $R_{\max} = 100$
- 损失函数：Normalized Focal Loss
- 优化器：Adam ($\beta_1=0.9, \beta_2=0.999$)

## 相关工作脉络
1. **RITM [27]**：开山之作，首次将历史掩码作为输入，但直接拼接未做调制，信息传递效率有限。
2. **BRS/f-BRS [14,26]**：推理时反向传播优化点击图或特征，计算成本高，MFP为纯前向推理。
3. **SimpleClick [21]**：轻量ViT骨干代表，MFP在其基础上引入调制机制显著提升性能。
4. **FocalClick [4] / FocusCut [19]**：局部窗口细化策略，MFP通过全局调制实现类似效果且无需裁剪。
5. **EMC-Click [7]**：自注意力+相关性模块传播点击信息，MFP以轻量调制替代复杂注意力计算。
6. **iCMFormer [16]**：跨模态ViT方法，MFP在相同ViT-B骨干下仍更优，证明调制机制有效性。

## 局限性与未来方向
- **仅支持单物体交互**：当前框架针对单目标分割设计，未扩展至多物体场景。
- **参数敏感性**：$\Gamma$、$R_{\max}$、$N$等超参数需人工设定，缺乏自适应机制。
- **点击策略依赖**：评测使用自动点击策略，真实用户交互行为可能与假设存在差异。
- **计算开销**：调制过程增加少量计算，但未量化与基线的推理延迟对比。

## 研究启发与可借鉴点
1. **历史信息调制思想可迁移**：概率图调制策略可推广至视频分割、医学图像交互标注等需要历史状态的任务。
2. **递归训练范式**：按预测误差递进采样的训练策略值得在其他序列决策任务中验证。
3. **Late Fusion设计**：在分割头前融合辅助特征而非仅作为输入通道，能更强引导最终预测，适用于多模态融合场景。
4. **距离度量自适应切换**：前期用语义距离、后期用空间距离的思路，可用于多阶段细化任务。
5. **轻量化验证完整性**：在ResNet-34/HRNet-18等轻量骨干上验证，证明方法不依赖大模型，工程实用性强。

## 关键术语表
**Interactive Image Segmentation**：通过用户交互标注（点击/框/草图）实现的目标分割任务。
**Probability Map**：网络输出的每个像素属于前景的概率图，反映当前预测置信度。
**Gamma Correction**：非线性变换技术，通过指数运算增强或抑制概率值，此处用于调制概率图。
**NoC (Number of Clicks)**：达到指定IoU阈值所需的平均点击次数，越低越好。
**Recursive Training**：按交互顺序递归采样点击并训练，充分利用历史概率图信息。
**Late Fusion**：在网络深层（分割头前）融合多源特征，区别于Early Fusion的输入层拼接。
**Modulation Window**：以当前点击为中心、半径R的圆形区域，仅在此范围内进行概率调制。
**Probability Distance**：基于概率值差异的距离度量，用于前期点击的伽马分配。

## 可复现要素
- **数据集**：GrabCut、Berkeley、DAVIS、SBD、COCO、LVIS均为公开数据集。
- **代码开源**：https://github.com/cwlee00/MFP
- **权重**：论文未明确提及预训练权重公开情况。
- **关键超参**：$N=7$、$R_{\max}=100$、$\Gamma$使点击处概率调至0.99/0.01。
- **训练细节**：Random resizing/cropping/flipping/rotation/brightness augmentation；Normalized Focal Loss；Adam优化器。
