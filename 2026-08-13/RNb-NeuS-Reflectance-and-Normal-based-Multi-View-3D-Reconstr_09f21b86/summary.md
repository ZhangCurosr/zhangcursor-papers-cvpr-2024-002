---
title: "RNb-NeuS-Reflectance-and-Normal-based-Multi-View-3D-Reconstr"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Brument_RNb-NeuS_Reflectance_and_Normal-based_Multi-View_3D_Reconstruction_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 15:21:02"
---

# 论文速读：RNb-NeuS-Reflectance-and-Normal-based-Multi-View-3D-Reconstr

## 一句话总结
本文提出了一种基于反射率与法向量联合重参数化的神经体积渲染（NVR）多视图3D重建方法（RNb-NeuS）。该方法将光度立体（PS）估计的异构输入统一映射为模拟任意光照下的辐射度向量，仅依赖单一目标优化即可无缝融合几何与光度信息，在DiLiGenT-MV基准上全面超越现有MVPS方法，尤其在高曲率与低可见度区域的细节恢复上取得突破。

## 研究问题与动机
- **MVS的非朗伯体局限**：传统多视图立体（MVS）依赖亮度一致性假设，在非朗伯材质场景下失效，且固定光照下难以恢复极细几何结构。
- **PS的低频几何缺陷**：光度立体擅长恢复高频法向与反射率细节，但单一视角下低频几何重建精度不足，且缺乏跨视角全局一致性约束。
- **现有MVPS的多目标冲突**：近期深度MVPS方法（如Kaya22/23、PS-NeRF）采用多损失联合优化，几何保真项与光度一致性项可能存在冲突，依赖不确定的超参数调优，易导致细微结构丢失。
- **缺乏统一灵活的集成范式**：现有方案多为特定PS网络定制端到端架构，难以通用化集成任意校准/非校准、经典/深度学习的PS前端，且优化目标复杂。

## 核心贡献（创新点）
1. **像素级联合重参数化**：将标量反射率与单位球法向量通过线性朗伯PBR模型映射为同构的辐射度向量。与直接联合优化异构几何/光度量不同，本质消除了目标间的维度与尺度冲突，使单一损失函数可同时约束两者。
2. **单目标NVR重建框架**：在NeuS体积渲染管线中去除视锥方向依赖，显式引入任意光照下的反照率场一致性匹配。与PS-NeRF等依赖多个MLP显式分解BRDF/光照/可见性的方案相比，大幅降低模型复杂度与优化难度。
3. **通用MVPS流水线设计**：构建包含PS前端输入、不确定性过滤、跨视角尺度对齐、最优光照三元组模拟、SDF优化的完整流程。与MVPSNet等特定网络不同，本框架即插即用，兼容任意现有或未来PS方法。
4. **高曲率与低可见度区域细节突破**：结合像素级最优光照配置与角度偏差阈值筛选，在极端几何与遮挡区域实现稳定重建。与Kaya23等基线相比，高曲率区域CD误差增幅仅4%（竞品最高96%），低可见度区域仅13%（竞品最高81%）。

## 方法详解
- **场景表征**：几何由隐式SDF场 $f: \mathbb{R}^3 \to \mathbb{R}$ 表示，表面为零水平集 $S=\{\mathbf{x}|f(\mathbf{x})=0\}$；光度由标量反照率场 $\rho: \mathbb{R}^3 \to \mathbb{R}$ 表示。
- **重参数化原理**：对每个像素选取3个线性无关的任意光照向量 $\mathbf{l}_{k,1}, \mathbf{l}_{k,2}, \mathbf{l}_{k,3}$，构造同质辐射度向量 $\mathbf{v}(\mathbf{n}_k, r_k) = r_k \mathbf{L}_k \mathbf{n}_k \in \mathbb{R}^3$。该映射在 $r_k \neq 0$ 且 $\mathbf{L}_k$ 非奇异时保持双射，原始反射率与法向量可无损还原。允许负辐射度（自阴影）以保持数学一致性。
- **体积渲染适配**：沿用NeuS的体积积分形式 $\tilde{v}_{k,l} = \int_{t_n}^{t_f} w(t, f(\mathbf{x}_k(t))) c_l(\mathbf{x}_k(t)) dt$，但将颜色函数改为 $c_l(\mathbf{x}) = \rho(\mathbf{x}) \nabla f(\mathbf{x})^\top \mathbf{l}_{k,l}$，使渲染值与输入重参数化向量在同一光照条件下可比。
- **单一优化目标**：整体损失仅含数据保真项与Eikonal正则化：
  $\min_{f,\rho} \sum_{k=1}^m \|\mathbf{v}(\mathbf{n}_k, r_k) - \tilde{\mathbf{v}}_k(f, \rho)\|_1 + \lambda \mathcal{L}_{\mathrm{reg}}(f)$
  其中 $\mathcal{L}_{\mathrm{reg}}(f) = \frac{\sum_k \int_{t_n}^{t_f} (\|\nabla f(\mathbf{x}_k(t))\|^2 - 1)^2 dt}{m(t_f - t_n)}$，无需多损失加权即可同步约束几何与光度一致性。
- **MVPS六步流程**：① SDM-UniPS计算各视角反射率/法向量（100次随机10图采样取中值）；② 按法向角度偏差阈值 $\tau=15^\circ$ 剔除不可靠像素；③ 跨视角中值比法对齐反射率尺度；④ 按局部法向动态选择最优光照三元组（倾角54.74°、方位差120°）生成模拟辐射度；⑤ NeuS架构优化300k次（batch 512像素）；⑥ Marching Cubes提取网格。

## 实验与结果
- **数据集**：DiLiGenT-MV（5个真实物体，20视角×96光照，含GT法向与曲率标注）。
- **评估指标**：Chamfer Distance (CD)、F-score、平均角度误差 (MAE)；额外细分高曲率区（|曲率|>1.6，占8.27%顶点）与低可见度区（<5视角可见，占8.70%顶点）。
- **基线对比**：Park1
