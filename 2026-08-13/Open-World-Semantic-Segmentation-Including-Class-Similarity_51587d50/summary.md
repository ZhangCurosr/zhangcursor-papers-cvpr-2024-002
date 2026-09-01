---
title: "Open-World-Semantic-Segmentation-Including-Class-Similarity"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Sodano_Open-World_Semantic_Segmentation_Including_Class_Similarity_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 20:14:31"
field: "开放世界语义分割"
keywords: ["open-world semantic segmentation", "anomaly segmentation", "novel class discovery", "feature space", "contrastive learning", "class similarity"]
innovations: ["双解码器架构：语义解码器学习类特征描述符+对比解码器执行异常分割", "特征损失函数：通过累积类原型均值和方差在特征空间约束同类像素", "高斯后处理+融合策略：利用类高斯分布似然和对比特征范数融合判定未知像素"]
benchmarks: ["SegmentMeIfYouCan", "BDDAnomaly"]
---

# 论文速读：Open-World-Semantic-Segmentation-Including-Class-Similarity

## 一句话总结
本文提出了一种名为 ContMAV 的双解码器卷积神经网络，在保持高精度封闭世界语义分割的同时，无需额外训练数据即可实现异常分割与新品类发现，并为每个新发现的未知类别提供与已知类别的相似度度量。

## 研究问题与动机
1. 自动驾驶等自主系统在真实世界中会遇到训练时未见过的对象，但现有封闭世界（closed-world）语义分割模型对未知对象会过度自信地错误分类为已知类别
2. 现有异常检测方法（如 softmax 阈值、熵最大化、能量分数、生成模型重建等）要么依赖不确定性估计，要么需要额外数据，要么无法区分不同未知类别
3. 下游任务（如路径规划、建图）需要知道"未知对象类似某已知类别"这一信息，以支持保守模式切换或动态物体追踪

## 核心贡献（创新点）
1. **双解码器网络架构**：一个语义解码器做封闭世界分割并学习类特征描述符，一个对比解码器做二进制异常分割；与以往依赖 softmax 阈值或生成模型的方法不同，本文直接在特征空间操作来区分已知/未知
2. **特征损失函数 $\mathcal{L}_{\mathrm{feat}}$**：通过累积训练过程中真阳性像素的 pre-softmax 特征来动态维护每类的均值 $\pmb{\mu}_k$ 和方差 $\sigma_k^2$，并将同类像素推向该类描述符；与以往仅优化分类损失的方法本质不同，该方法在特征空间构建了每类的高斯分布模型
3. **对比解码器结合对比损失与 Objectosphere 损失**：将已知类特征推至半径为 $\xi$ 的超球面表面，将未知类特征压向球心；不同于以往依赖不确定性估计或额外 OoD 数据的方法，该方法无额外数据需求
4. **高斯后处理 + 新品类发现机制**：测试时用高斯似然评分判断像素是否未知，并通过距离阈值 $\eta$ 聚合为新品类；与 K-means 等后处理方法不同，该方法无需重新聚类全部特征，支持增量式新品类发现
5. **类别相似度度量**：通过高斯得分 $\arg\max_k s_k(f_p)$ 输出未知对象最相似的已知类别；区别于仅看最大激活值的方法，该度量考虑了特征分布的协方差信息

## 方法详解
**网络架构**：采用 ResNet34 编码器（Basic Block 替换为 NonBottleneck-1D 轻量块），后接金字塔池化模块（PSP）扩展感受野；两个解码器均使用 SwiftNet 模块 + 最近邻上采样，通过 skip connections 融合多尺度特征。

**语义解码器损失**：
- 加权交叉熵 $\mathcal{L}_{\mathrm{sem}}$：标准分割损失，权重 $\omega_k$ 按类别逆频率计算
- 特征损失 $\mathcal{L}_{\mathrm{feat}}$（Eq.5）：将每个 GT 为 $k$ 的像素的 pre-softmax 特征 $f_p$ 推向上一 epoch 累积的类均值 $\pmb{\mu}_k^{e-1}$，并以方差 $\sigma_k^2$ 归一化（第 1 个 epoch 不生效）
- 总损失 $\mathcal{L}_{\mathrm{sdec}} = w_1 \mathcal{L}_{\mathrm{sem}} + w_2 \mathcal{L}_{\mathrm{feat}}$

**类原型累积**（Eq.3-4）：每 epoch 收集真阳性像素的 $f_p$ 计算均值 $\pmb{\mu}_k$ 和逐元素方差 $\sigma_k^2$，作为该类别的高斯分布参数。

**对比解码器损失**：
- 对比损失 $\mathcal{L}_{\mathrm{cont}}$（Eq.8）：对图像内每类求均值特征 $\overline{f}_k$，使其与上一 epoch 归一化均值 $\bar{\pmb{\mu}}_k$ 相似度最大化、与其他类均值区分
- Objectosphere 损失 $\mathcal{L}_{\mathrm{obj}}$（Eq.9）：已知类像素特征范数 $\geq \xi$，未知类（void）像素特征范数 $\to 0$
- 总损失 $\mathcal{L}_{\mathrm{cdec}} = w_3 \mathcal{L}_{\mathrm{cont}} + w_4 \mathcal{L}_{\mathrm{obj}}$

**后处理（测试阶段）**：
- 语义侧高斯评分（Eq.11-13）：对像素 $p$ 预测为类 $k$ 时计算 $s_k(f_p) = \exp(-\frac{1}{2}(f_p-\pmb{\mu}_k)^\top \Sigma_k^{-1}(f_p-\pmb{\mu}_k))$，取 $\max_k$ 得 $s(p)$，未知分数 $s_{\mathrm{unk},p}^{\mathrm{sem}} = 1 - s(p)$
- 对比侧分数（Eq.13）：$s_{\mathrm{unk},p}^{\mathrm{cont}} = \max(0, 1 - \|f_p\|^2/\xi)$
- 融合（Eq.14）：$s_{\mathrm{unk},p} = \frac{1}{2}(s_{\mathrm{unk},p}^{\mathrm{sem}} + s_{\mathrm{unk},p}^{\mathrm{cont}})$，超过阈值 $\delta$ 判定为未知
- 新品类发现：未知像素按特征距离阈值 $\eta$ 聚合到已有新品类原型 $\mathcal{F}$ 或创建新类

**类别相似度**：取 $\tilde{k} = \arg\max_k s_k(f_p)$ 作为该未知像素最相似的已知类别。

## 实验与结果
**数据集**：SegmentMeIfYouCan（Cityscapes 训练 + 合成未知）、BDDAnomaly（含可获取的测试集 GT）。

**主要结果**：
- 封闭世界 mIoU：Cityscapes 70.8%（OW vs CW 71.1%，几乎无损），BDDAnomaly 62.8%（vs CW 64.1%）
- SegmentMeIfYouCan 公开排行榜：**FPR95 = 3.8%（第 1）**、**PPV = 61.9%（第 1）**、**mean F1 = 63.6%（第 1）**；AUPR 90.2%（第 6）、sIoU 54.5%（第 6）。相比 Maskomaly（无 OoD 数据）：FPR95 从 6.9% 降至 3.8%（提升 45%），mean F1 从 49.9% 提升至 63.6%
- BDDAnomaly：AUPR 96.1%、FPR95 6.9%，优于 MaxSoftmax（3.7%/24.5%）和 RbA（15.9% FPR95）
- 新品类发现（BDDAnomaly）：Train mIoU 62.4%、Motorcycle 62.2%、Bicycle 56.8%；移除特征损失后分别降至 48.1%/53.8%/39.9%，证明特征损失的关键作用
- 类别相似度（BDDAnomaly*）：Motorcycle→Car 准确率 58.9%、Train→Truck 准确率 49.9%，远超线性分类器基线（12.5%/9.8%）

**消融结论**：
- 最优异常分割组合：$\mathcal{L}_{\mathrm{feat}}$ + 对比解码器 + Gaussian 后处理（K 行，AUPR 96.1%，FPR95 6.9%）
- Gaussian 后处理（J/K）优于最小距离法（H/I）和 softmax 阈值法（A/B/C/D/E/F）
- 对比解码器单独（G）已优于单纯 softmax 阈值（A），但与特征损失结合后显著提升（K vs D/F）

## 相关工作脉络
1. **softmax 阈值/背景类方法**（MaxSoftmax [22]、Background [6]）：仅依赖输出层概率，对未知样本仍高度自信，无法区分不同未知类；本文在特征空间操作，从根本上缓解 overconfidence
2. **熵最大化/能量分数方法**（[15], [35]）：要求 CNN 输出本身具备不确定性感知能力；本文不依赖此假设，而是通过对比解码器显式构造已知/未知特征分布
3. **生成/重建方法**（Maskomaly [1], DRaEM [67]）：需训练生成模型并利用重建误差；本文无需生成模型，训练更高效
4. **视觉-语言模型方法**（WinCLIP [25], DenseCLIP [48]）：依赖大规模预训练；本文完全在分割特征空间学习，无需外部预训练
5. **OoD 数据增强方法**（DenseHybrid [19], RbA [44]）：需额外标注的分布外数据；本文仅需 Cityscapes 训练，无额外数据
6. **闭集强分类器方法**（OpenSetNet [2], Open-Set Recognition [60]）：针对分类任务；本文将其思想迁移至像素级分割，并提出特征损失与高斯后处理以适应密集预测

## 局限性与未来方向
1. **超参数敏感**：阈值 $\delta$、$\eta$、$\xi$ 及损失权重需手动调节，缺乏自适应机制
2. **特征空间学习依赖充足训练数据**：在高基数类别或分布偏移严重时，类原型累积可能不稳定
3. **对小目标异常检测受限**：BDDAnomaly 中自行车/摩托车像素占比极小时性能下降明显（作者附录提及）
4. **不支持增量学习**：新品类通过离线聚合发现，无法在线持续学习新类别
5. **未来方向**：与 CLIP 等视觉-语言模型结合利用语义先验；扩展到视频流时序异常检测；扩展到 RGB-D 或多模态输入；引入自适应阈值替代手工调参

## 研究启发与可借鉴点
1. **类原型驱动的 feature loss**：将每类真阳性像素的 pre-softmax 特征累积为均值和方差，并以此作为训练时的正则化目标——这一技巧可迁移至任何需要在特征空间建模类分布的任务（如开放集识别、Few-shot 分割）
2. **对比解码器 + Objectosphere 损失的组合设计**：将已知类推至超球面、未知类压向原点，是一种无需额外数据的"显式边界"策略，可迁移至其他异常检测/开放集分类任务
3. **高斯后处理用于开放世界判定**：用逐像素到高斯分布的 Mahalanobis 距离替代简单的 softmax 阈值，显著提升了异常判定的可靠性；这一后处理范式可独立应用于各类分割模型
4. **双解码器共享 encoder 的解耦设计**：语义解码器保精度，对比解码器专责异常感知，两者通过共享特征相互增强——这一架构思想适用于需要同时完成多项相关任务的场景
5. **新品类发现通过距离阈值聚合**：无需 K-means 重聚类，通过增量式原型匹配实现"无限"新品类发现，计算高效且适合流式部署

## 关键术语表
**Open-World Semantic Segmentation**：在测试图像中包含训练时未见类别的场景下，同时完成已知类精确分割与未知区域检测的像素级理解任务
**Anomaly Segmentation（异常分割）**：将图像像素二分类为"已知"或"未知/异常"，是开放世界分割的基础子任务
**Novel Class Discovery（新品类发现）**：在仅有已知类别标注的前提下，将未知样本像素级聚合成不同新品类的任务
**Feature Space Manipulation（特征空间操作）**：直接在网络中间层特征而非输出概率上进行正则化或约束，以实现更好的分布分离
**Objectosphere Loss（物体空间损失）**：将已知类特征约束到超球面表面、将未知类特征压向球心的损失函数，用于构造明确的已知/未知边界
**Mean Activation Vector（均值激活向量，MAV）**：每类所有真阳性像素 pre-softmax 特征的滑动平均，作为该类的类描述符/原型
**Gaussian Post-processing（高斯后处理）**：利用训练阶段累积的类均值和方差构建高斯模型，在测试时通过 Mahalanobis 似然评分判定像素是否属于已知类

## 可复现要素
- **数据集**：SegmentMeIfYouCan（公开，需注册获取）；BDDAnomaly（公开）；BDDAnomaly*（作者附录说明，基于 BDDAnomaly 修改）
- **代码/权重**：论文未提供开源链接（论文脚注标注为"our approach¹"，¹号注释在 PDF 中未显示，可能为会议盲审期处理方式），需联系作者获取
- **关键超参**：$\xi = 1$，$\delta = 0.6$，$\tau = 0.1$，$\eta = 0.5$；$w_1 = 0.9, w_2 = 0.1, w_3 = 0.5, w_4 = 0.5$；学习率 0.004（one-cycle policy）；batch size 8；500 epochs；Adam 优化器；Random scale/crop/flip 数据增强
- ** backbone**：ResNet34（NonBottleneck-1D 替换 BasicBlock）+ Pyramid Pooling Module + SwiftNet decoder
