---
title: "Open-World-Semantic-Segmentation-Including-Class-Similarity"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Sodano_Open-World_Semantic_Segmentation_Including_Class_Similarity_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 00:44:06"
field: "开放世界视觉感知"
keywords: ["开放世界语义分割", "异常分割", "新类别发现", "对比学习", "特征空间建模", "类相似度"]
innovations: ["双解码器架构联合优化闭世界分割与异常检测", "特征损失函数约束类原型分布实现特征空间显式建模", "基于高斯分布的类相似度度量无需额外训练"]
benchmarks: ["SegmentMeIfYouCan", "BDDAnomaly"]
---

# 论文速读：Open-World-Semantic-Segmentation-Including-Class-Similarity

## 一句话总结
本文提出了一种基于双解码器CNN的开放世界语义分割方法，能够在无需额外训练数据的情况下，同时实现高精度的闭世界语义分割、未知类别的异常分割（anomaly segmentation）以及新类别间的区分和相似度度量。

## 研究问题与动机
- **闭世界假设的局限**：传统语义分割模型在遇到训练时未见过的物体时，往往过度自信地将其分类为已知类别之一，无法识别"未知"。
- **开放世界场景需求**：自动驾驶等实际应用中，系统必须能够处理训练数据中未覆盖的罕见场景和物体，以增强鲁棒性。
- **多任务联合需求**：现有方法通常仅关注异常分割（二分类：已知/未知），缺乏对多个不同未知类别的区分能力，也无法提供新类别与已知类别的相似度信息，不利于下游规划或建图任务。
- **不依赖额外数据的挑战**：许多现有方法需要额外的out-of-distribution数据或生成模型进行训练，增加了复杂性和成本。

## 核心贡献（创新点）
- **双解码器架构**：提出一个语义解码器（负责闭世界分割和特征空间约束）与一个对比解码器（负责异常分割）相结合的轻量级网络，与现有单解码器方法相比实现了多任务联合优化。
- **特征损失函数（$\mathcal{L}_{\text{feat}}$）**：引入基于运行平均类原型和方差的特征损失，将同类像素的特征推向高斯分布中心，使特征空间具有明确的类边界，区别于仅依赖softmax输出的传统方法。
- **对比解码器与混合损失**：结合contrastive loss和objectosphere loss，将已知类特征分布在超球面上、未知类特征推向球心，实现无需额外训练数据的异常分割。
- **类相似度度量机制**：利用高斯分布得分作为相似度依据，为每个未知类别自动匹配最相似的已知类别，这是现有方法中较少提供的辅助信息。
- **多数据集SOTA性能**：在SegmentMeIfYouCan基准上取得FPR95第一、PPV第一、mean F1第一的成绩，BDDAnomaly上全面超越现有基线。

## 方法详解
- **网络架构**：基于ResNet34编码器（用NonBottleneck-1D块替换标准ResNet块以实现轻量化），后接金字塔池化模块（Pyramid Pooling Module）扩大感受野，然后通过两个SwiftNet解码器分别处理语义分割和异常检测。
- **语义解码器（Semantic Decoder）**：
  - 使用加权交叉熵损失$\mathcal{L}_{\text{sem}}$进行标准语义分割。
  - 引入特征损失$\mathcal{L}_{\text{feat}}$：每个epoch累积正确预测像素的预softmax特征作为类均值$\pmb{\mu}_k$和方差$\pmb{\sigma}_k^2$，损失函数强制当前epoch同类像素特征逼近上一epoch的均值。
  - 总损失：$\mathcal{L}_{\text{sdec}} = w_1 \mathcal{L}_{\text{sem}} + w_2 \mathcal{L}_{\text{feat}}$。
- **对比解码器（Contrastive Decoder）**：
  - Contrastive loss：使类均值特征$\overline{\pmb{f}}_k$与上一epoch的均值$\bar{\pmb{\mu}}_k^{e-1}$相似，同时与其他类区分。
  - Objectosphere loss：已知类特征范数大于阈值$\xi$，未知类特征范数趋近于0。
  - 总损失：$\mathcal{L}_{\text{cdec}} = w_3 \mathcal{L}_{\text{cont}} + w_4 \mathcal{L}_{\text{obj}}$。
- **后处理与开放世界分割**：
  - 语义解码器部分：为每类构建多维高斯分布$N(\pmb{\mu}_k, \pmb{\Sigma}_k)$，通过二次指数核计算拟合得分$s_k(\pmb{f}_p)$，得分低则判定为未知。
  - 对比解码器部分：通过特征范数与阈值$\xi$比较得到未知得分。
  - 两路得分加权融合：$s_{\text{unk},p} = \frac{1}{2}(s_{\text{unk},p}^{\text{sem}} + s_{\text{unk},p}^{\text{cont}})$，超过阈值$\delta$判定为未知。
- **未知类别发现与聚类**：未知像素的特征向量与已发现未知类均值距离小于阈值$\eta$时归入同一类，否则创建新类。
- **类相似度**：取高斯得分最高（即距离均值最近）的已知类作为最相似类别。

## 实验与结果
- **数据集**：SegmentMeIfYouCan（Cityscapes训练，含未知类别）、BDDAnomaly、BDDAnomaly*（用于类相似度评估）。
- **闭世界性能**：Open-world方法在Cityscapes和BDDAnomaly上的mIoU分别为70.8%和62.8%，与闭世界模型（71.1%、64.1%）接近，证明未知检测不影响已知类分割精度。
- **异常分割（SegmentMeIfYouCan）**：FPR95为3.8%（第一），PPV为61.9%（第一），mean F1为63.6%（第一），AUPR为90.2%（第六）。优于DenseHybrid、RbA、Maskomaly等基线。
- **异常分割（BDDAnomaly）**：AUPR为96.1%，FPR95为6.9%，全面超越MaxSoftmax、Background、MC Dropout等基线。
- **开放世界语义分割（BDDAnomaly）**：ContMAV（带特征损失）在Train/Motorcycle/Bicycle上的mIoU分别为62.4%/62.2%/56.8%，优于无特征损失版本（48.1%/53.8%/39.9%）和Baseline（0%/32.3%/32.8%）。
- **类相似度（BDDAnomaly*）**：ContMAV在Motorcycle（最似car）和Train（最似truck）上的准确率分别为58.9%和49.9%，优于仅用最大激活（39.9%/27.6%）和专用分类器（12.5%/9.8%）。
- **消融实验**：高斯后处理（Gs）策略在所有指标上达到最佳；特征损失对未知类别区分和类相似度均有显著提升。

## 相关工作脉络
- **Softmax阈值法**（MaxSoftmax、Background）：直接对softmax输出阈值化，但闭世界模型对未知样本过于自信，效果差。本文在特征空间操作，不依赖softmax置信度。
- **能量熵/不确定性方法**（MC Dropout、Confidence、Energy-based）：依赖模型不确定性估计，需要额外推理开销。本文通过特征范数和距离度量实现确定性的异常检测。
- **生成模型方法**（Maskomaly、Draem、OmniAL）：利用重建误差检测异常，需要额外训练数据或复杂生成器。本文无需OoD数据和生成模型。
- **视觉语言模型**（CLIP-based如WinCLIP、DenseCLIP）：利用大规模预训练模型，计算成本高。本文方法轻量且仅依赖给定训练数据。
- **已有开放世界分割**（DenseHybrid、RbA）：部分使用额外OoD数据训练。本文完全不依赖额外数据，且在SegmentMeIfYouCan基准上取得更好性能。
- **对比学习在异常检测中的应用**（ContrastiveMAV相关工作）：本文结合contrastive loss与objectosphere loss，并将两者与特征损失结合用于像素级分割任务，是方法上的重要扩展。

## 局限性与未来方向
- **阈值敏感性**：未知判定阈值$\delta$和聚类距离阈值$\eta$需要手动调整，对不同类型场景的泛化性有待验证。
- **高斯分布假设**：假设每类特征服从对角度分布（独立维度），可能无法完全捕捉复杂的多模态特征分布。
- **仅适用于RGB图像**：当前方法仅使用RGB输入，未探索多模态（如RGB-D、LiDAR）的开放世界分割。
- **小目标检测挑战**：BDDAnomaly中摩托车/自行车等小目标检测仍然困难，说明对小尺度未知物体的感知能力有限。
- **未来可结合**：与视觉语言模型（如CLIP）结合以利用语义先验；扩展至视频序列的开放世界分割；探索自监督预训练对特征空间结构化的帮助。

## 研究启发与可借鉴点
- **特征空间显式建模**：通过类原型的均值和方差在特征空间建立明确的类边界，而非仅依赖输出层概率，这一思路可迁移至其他开放集识别任务。
- **双解码器解耦设计**：将闭世界分割和异常检测解耦为两个解码器，通过共享编码器协同训练，既保证各任务性能又避免单一网络的妥协。
- **迭代式类原型更新**：使用上一epoch的统计量引导当前epoch训练，类似"蒸馏"机制，避免梯度冲突，这一策略可用于持续学习或在线更新场景。
- **多维度后处理融合**：结合概率得分（高斯拟合）和几何度量（特征范数）进行综合决策，提升了鲁棒性，值得在其他异常检测任务中借鉴。
- **类相似度作为副产品**：利用高斯分布得分自然导出相似度度量，无需额外训练，可服务于规划、建图等下游应用。

## 关键术语表
- **Open-World Semantic Segmentation（开放世界语义分割）**：在测试时能够识别并分割训练阶段未见过的类别的语义分割任务。
- **Anomaly Segmentation（异常分割）**：将图像像素二分类为"已知"或"未知/异常"，不涉及未知类别的具体分类。
- **Novel Class Discovery（新类别发现）**：在已知类别训练的基础上，将未知像素聚类为不同的新类别。
- **Objectosphere Loss（超球面损失）**：约束已知类特征范数大于阈值、未知类特征范数趋近于零的损失函数。
- **Contrastive Loss（对比损失）**：拉近同类特征、推远异类特征的损失，常用于度量学习。
- **Feature Loss（特征损失）**：本文提出的损失，强制同类像素的特征逼近类均值并受方差约束。
- **Gaussian Post-Processing（高斯后处理）**：基于每类特征的高斯分布计算像素属于该类的得分，用于未知检测。
- **SegmentMeIfYouCan**：专为异常分割设计的benchmark，包含Cityscapes风格的图像和人工添加的未知物体。

## 可复现要素
- **数据集**：SegmentMeIfYouCan（公开）、BDDAnomaly（公开）、BDDAnomaly*（公开）。
- **代码/权重**：论文未明确提及代码是否开源（需查看项目页面确认）。
- **关键超参**：$\xi = 1$、$\delta = 0.6$、$\tau = 0.1$、$\eta = 0.5$；学习率0.004；batch size 8；500 epochs；loss权重$w_1=0.9, w_2=0.1, w_3=0.5, w_4=0.5$。
- **骨干网络**：ResNet34（NonBottleneck-1D块）、Pyramid Pooling Module、SwiftNet解码器。
- **训练策略**：One-cycle learning rate policy、随机缩放/裁剪/翻转数据增强、Adam优化器。
