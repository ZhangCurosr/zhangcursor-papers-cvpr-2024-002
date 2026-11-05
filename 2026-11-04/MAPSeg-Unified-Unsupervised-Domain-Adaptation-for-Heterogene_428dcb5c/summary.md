---
title: "MAPSeg-Unified-Unsupervised-Domain-Adaptation-for-Heterogene"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_MAPSeg_Unified_Unsupervised_Domain_Adaptation_for_Heterogeneous_Medical_Image_Segmentation_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 00:42:58"
---

# 论文速读：MAPSeg-Unified-Unsupervised-Domain-Adaptation-for-Heterogeneous

## 一句话总结
本文提出MAPSeg，一个统一的无监督域自适应（UDA）框架，通过融合3D掩码自编码（MAE）与掩码伪标签（MPL）及全局-局部特征协同（GLC），有效解决医学图像分割中的跨序列、跨站点、跨年龄与跨模态四类域偏移，并首次在同一架构下兼容集中式、联邦式与测试时UDA场景。

## 研究问题与动机
- 医学影像专家标注成本高昂且多中心/纵向研究中目标域标签往往不可得，导致源域训练的分割模型在面对域偏移时性能骤降。
- 现有UDA方法多针对单一偏移类型设计（如仅跨模态或仅跨序列），基于图像翻译的方法在处理跨年龄/跨序列偏移时会引入尺寸或结构失真，泛化性有限。
- 严格的隐私法规（HIPAA/GDPR）限制了多机构数据集中共享，传统集中式UDA难以直接落地，亟需支持联邦与测试时异步适配的统一方案。
- 部分前作UDA依赖目标域标签进行模型验证与早停，这在真实无标签场景中不现实，缺乏可靠的无监督评估信号。

## 核心贡献（创新点）
1. **统一MAE+MPL框架**：首次将3D多尺度掩码自编码与掩码伪标签系统性地结合，可同时应对医学分割中的四种主要域偏移；与以往仅针对单一偏移或依赖图像翻译的方法本质不同。
2. **全局-局部特征协同（GLC）模块**：利用医学图像解剖分布先验，在潜空间拼接局/全局特征并施加余弦相似度正则，有效抑制强域偏移下的伪标签漂移；与既往仅在输入/特征层做分布对齐的方法本质不同。
3. **三范式统一扩展**：将MAPSeg无缝适配至集中式、联邦式与测试时UDA，打破对同步集中数据的依赖；与以往联邦或测试时方法仅支持单一设定且性能落差大的工作本质不同。
4. **无目标标签的模型选择策略**：提出基于源域Dice与目标域伪标签损失加权组合的Score指标，在不使用目标域真值的情况下实现稳定ckpt选取，比依赖目标标签验证的方法更具临床实用性。

## 方法详解
- **3D多尺度MAE预训练**：采用3D CNN作为编码器，同时处理局部随机裁剪Patch（尺寸$8^3$，随机掩码70%）与全图下采样版本（尺寸$4^3$，对应更大视野）。以掩码区域像素重建的均方误差为目标，使编码器学会从可见Patch推断缺失区域的解剖语义。
- **3D掩码伪标签（MPL）**：构建教师-学生网络。教师模型$f_\theta$对完整目标图像$x_t$生成伪标签（梯度 detached）；学生模型$f_\phi$在掩码目标Patch $x_t^M$与非掩码源Patch $x_s^{\overline{M}}$上分别拟合伪标签与真实标注。损失为 $\mathcal{L}_{MPL} = \mathcal{L}_{Seg}(f_\phi(x_t^M), f_\theta(x_t)) + \beta \mathcal{L}_{Seg}(f_\phi(x_s^{\overline{M}}), y_s)$，$\beta=0.5$。教师参数通过EMA更新：$\theta_{t+1} = \alpha \theta_t + (1-\alpha)\phi_t$（大规模预训练$\alpha$取0.999→0.9999；小规模取0.99→0.999→0.9999）。
- **全局-局部协同（GLC）**：编码器输出局部特征$\chi_{loc}=g(x)$与从全局特征$g(X)$按掩码位置裁剪并上采样的$\chi_{glo}$进行通道拼接，由解码器$h$生成预测。引入余弦相似度正则 $\mathcal{L}_{cos}$ 促使局/全局表征一致，并在全局分辨率上施加辅助监督，防止解码器过度依赖局部纹理。源/目标域GLC损失分别由$\gamma=0.05$与$\delta=0.025$加权。
- **统一损失与场景扩展**：集中式总损失 $\mathcal{L} = \mathcal{L}_{FSS} + \mathcal{L}_{MPL} + \mathcal{L}_{GLC}$。联邦UDA将其解耦为服务器端的有监督损失（Eq.9）与客户端的自监督损失（Eq.10），通过FedAvg聚合教师EMA参数。测试时UDA进一步解除同步约束，源域Warm-up后分发模型，目标域独立运行MPL自训练。
- **无目标标签验证Score**：$Score = Dice_{Src} - 0.5 \times \overline{\mathcal{L}_{Seg}}(f_\phi(x_t^M), f_\theta(x_t))$，理论上限1.5。不使用目标真值即可逼近依赖标签验证的性能（差异仅0.9 Dice）。

## 实验与结果
- **数据集**：私有婴儿脑MRI（2421例，2306例无标注用于预训练，115例专家标注）；公开心脏CT-MRI（MMWHS 2017，40例，20 CT源 + 20 MRI目标）。
- **评估基线**：集中式（AdvEnt, DAFormer, HRDA, MIC, DAR-UNet）；联邦式（FAT, DualAdapt）；跨模态（PnP-AdaNet, SIFA-V1/V2, DAFormer, MPSCL, MA-UDA, SE-ASA, FSUDA-V1/V2, PUFT, SDUDA）。
- **核心结果**：
  - 集中式UDA：婴儿脑MRI跨序列/跨站点/跨年龄平均Dice分别为77.7%、73.1%、75.2%，较前SOTA（DAR-UNet/MIC等）分别提升10.5、~10、~10个点；在心脏CT→MRI任务上平均Dice达81.2%，超越SDUDA（74.8）等所有公开基线，提升5.7个点。
  - 联邦UDA（Fed-MAPSeg）：跨序列/跨站点/跨年龄Dice为69.9%/73.6%/71.0%，大幅领先FAT（27.6/63.8/69.0）与DualAdapt，且与集中式性能接近。
  - 测试时UDA：目标域性能下降<3%，源域因单站独立适配出现轻微遗忘
