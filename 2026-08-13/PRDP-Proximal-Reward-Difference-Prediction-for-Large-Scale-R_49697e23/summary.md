---
title: "PRDP-Proximal-Reward-Difference-Prediction-for-Large-Scale-R"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Deng_PRDP_Proximal_Reward_Difference_Prediction_for_Large-Scale_Reward_Finetuning_of_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 20:14:48"
field: "扩散模型奖励微调"
keywords: ["diffusion models", "reward finetuning", "RLHF", "black-box optimization", "proximal policy optimization", "generative alignment"]
innovations: ["将RLHF目标转化为等价的监督回归目标RDP，消除策略梯度方差", "设计近端裁剪与在线快照更新机制实现10万+prompt稳定训练"]
benchmarks: ["HPDv2", "Pick-a-Pic v1", "HPSv2", "PickScore"]
---

# 论文速读：PRDP-Proximal-Reward-Difference-Prediction-for-Large-Scale-R

## 一句话总结
本文提出PRDP（Proximal Reward Difference Prediction），首次实现了扩散模型在超过10万提示词的大规模数据集上的稳定黑盒奖励微调，通过将RLHF目标转化为等价的监督回归目标，解决了传统策略梯度方法在大规模训练中的不稳定性问题。

## 研究问题与动机
- **扩散模型训练目标与下游应用不对齐**：扩散模型的极大似然训练目标难以直接适配生成美学偏好图像或新颖对象组合等下游任务。
- **视觉领域RLHF迁移受阻**：语言领域RLHF的成功启发了视觉奖励微调，但视觉域的RL方法面临策略梯度高方差导致的训练不稳定问题。
- **现有RL微调方法无法扩展至大规模**：DDPO在400个prompt上表现稳定，DPOK在200个prompt上有效，但两者在超过10万prompt时均出现训练崩溃或性能退化。
- **黑盒奖励场景的通用性需求**：许多实用奖励模型（如PickScore、HPSv2）是不可微的，需要不依赖梯度回传的微调方法。

## 核心贡献（创新点）
- **提出RDP监督回归目标**：将RLHF的KL正则化奖励最大化目标重新表述为预测去噪轨迹奖励差的MSE回归任务，并严格证明两者最优解等价（$\pi_\theta = \pi_{\theta^*} \iff \mathcal{L}(\theta)=0$）。
- **与DDPO/DPOK的本质区别**：DDPO/DPOK依赖策略梯度（REINFORCE/PPO），方差随prompt数量急剧放大；PRDP完全消除策略梯度，转为确定性梯度下降，从根本上消除训练不稳定性。
- **设计在线优化与近端更新机制**：采用类似PPO的近端裁剪策略限制每步参数更新幅度，并结合在线采样（每K步更新参考模型快照），兼顾训练稳定性与样本分布覆盖。
- **首次实现10万+ prompt稳定微调**：在HPDv2和Pick-a-Pic v1的大规模实验中，PRDP保持稳定训练并显著提升生成质量，而DDPO完全失效。

## 方法详解
- **RDP目标推导**：从KL正则化奖励最大化目标出发，推导最优解满足 $\log \frac{\pi_{\theta^*}(\bar{x}|c)}{\pi_{ref}(\bar{x}|c)} = \frac{1}{\beta}r(x_0,c) - \log Z(c)$。通过对同一prompt的两个去噪轨迹取差分消去配分函数 $Z(c)$，得到 $\Delta \hat{r}_\theta(\bar{x}^a, \bar{x}^b, c) = \Delta r(x_0^a, x_0^b, c)/\beta$。
- **损失函数**：$\mathcal{L}(\theta) = \mathbb{E}_{\bar{x}^a, \bar{x}^b, c}\|\Delta \hat{r}_\theta(\bar{x}^a, \bar{x}^b, c) - \Delta r(x_0^a, x_0^b, c)/\beta\|^2$，其中 $\hat{r}_\theta(\bar{x}, c) = \log \frac{\pi_\theta(\bar{x}|c)}{\pi_{ref}(\bar{x}|c)}$。
- **近端更新（Proximal Updates）**：借鉴PPO的clipping机制，将 $\hat{r}_{\theta,t}(\bar{x}, c) = \log \frac{\pi_\theta(x_{t-1}|x_t, c)}{\pi_{ref}(x_{t-1}|x_t, c)}$ 裁剪到 $[\hat{r}_{\theta_{old},t} - \epsilon, \hat{r}_{\theta_{old},t} + \epsilon]$，最终损失取clip版本与原版本的max，防止单步更新过大。
- **在线优化（Online Optimization）**：每epoch开始前快照 $\pi_{\theta_{old}} \leftarrow \pi_\theta$，用 $\theta_{old}$ 采样去噪轨迹并计算奖励，每K步梯度更新后刷新快照；相比离线固定数据集采样，能持续追踪当前模型分布。
- **算法流程**：Algorithm 1描述了完整训练过程：初始化、采样prompt、用旧模型采样轨迹对、计算奖励、内层梯度更新、周期刷新快照。

## 实验与结果
- **小规模实验（45个prompt）**：PRDP与DDPO在HPSv2和PickScore奖励下表现相当，PRDP略优（HPSv2: 0.3471 vs 0.3398；PickScore: 0.2700 vs 0.2664）。
- **大规模实验（HPDv2 >10万prompt）**：PRDP在seen prompts上HPSv2达0.3175（DDPO仅0.2464），在unseen prompts（Pick-a-Pic v1测试集、HPDv2四类子集）上全面领先，DDPO甚至低于原始SD v1.4（Photo类别仅0.2093 vs SD的0.2750）。
- **多奖励微调（Pick-a-Pic v1）**：加权组合PickScore=10、HPSv2=2、Aesthetic=0.05，PRDP在复杂未见prompt上生成质量显著优于SD v1.4和DDPO。
- **在线vs离线优化**：在线优化在约10个epoch内即匹配离线优化性能，并持续提升；离线优化受限于固定数据集分布。
- **KL正则化防reward hacking**：使用LAION aesthetic predictor时，DDPO无KL正则化会忽略文本条件生成同质图像；PRDP（$\beta=10$）能同时提升美学质量并保持文本对齐。

## 相关工作脉络
- **DDPO (Black et al., 2024)**：基于PPO的扩散模型RL微调，维持每prompt独立缓冲区估计奖励均值/方差，在400 prompt上稳定但大规模失效；PRDP从根本上避免策略梯度，可扩展至10万+ prompt。
- **DPOK (Fan et al., 2023)**：通过拟合值函数降低策略梯度方差，在200 prompt上有效但性能不如DDPO；同样受限于大规模不稳定性。
- **DPO (Rafailov et al., 2023)**：语言模型中的监督偏好优化，将RL目标转化为分类目标；本文受其启发但重新推导出适用于扩散模型轨迹分布的回归形式。
- **DRaFT (Clark et al., 2024)**：直接对可微奖励回传梯度微调扩散模型；本文处理不可微黑盒奖励，适用性更广。
- **Diffusion-DPO (Wallace et al., 2024)**：并发工作，将DPO适配到大模型离线偏好数据；本文聚焦在线奖励最大化与训练稳定性。
- **RAFT (Dong et al., 2023)**：奖励排序微调，依赖离线高奖励样本；本文通过在线采样持续探索新分布。

## 局限性与未来方向
- **计算开销较高**：每步需采样多对去噪轨迹并计算奖励，大规模训练的显存和时间成本显著高于基础微调。
- **超参数敏感**：KL正则化强度$\beta$和近端裁剪范围$\epsilon$需手动调优，缺乏自适应策略。
- **奖励模型质量依赖**：方法效果受限于reward model的准确性，若reward model存在系统性偏差，微调结果可能放大该偏差。
- **未探索更高效的采样策略**：当前使用固定步数去噪采样，未来可结合加速采样方法降低每步成本。

## 研究启发与可借鉴点
- **RL→监督学习的范式转换**：将高方差的策略梯度目标转化为等价的监督回归目标，是提升训练稳定性的通用思路，可迁移至其他生成模型的奖励优化场景。
- **在线快照更新机制**：每K步刷新参考模型采样分布，既避免分布漂移又保持样本多样性，可借鉴于其他在线微调方法。
- **近端裁剪保护生成质量**：将PPO的clip技巧适配到扩散模型的轨迹对数概率比上，防止reward hacking同时保持文本对齐，为多目标权衡提供可行方案。
- **配对差值消除配分函数**：通过对同一条件采样两个轨迹取差分来消去不可计算的归一化常数，是处理扩散模型分布对比的优雅技巧。

## 关键术语表
**RDP (Reward Difference Prediction)**：将RLHF目标转化为预测去噪轨迹奖励差的监督回归目标，与原始RL目标最优解等价。

**PRDP (Proximal RDP)**：在RDP基础上引入近端裁剪和在线优化，实现大规模稳定微调的完整方法。

**黑盒奖励 (Black-box Reward)**：仅可通过前向查询获取评分、无法回传梯度的奖励函数，如PickScore和HPSv2。

**近端更新 (Proximal Update)**：借鉴PPO的clipping机制，限制模型每步参数更新幅度以防止训练崩溃。

**在线优化 (Online Optimization)**：使用当前模型快照而非固定离线数据采样去噪轨迹，使训练分布持续跟随模型更新。

**KL正则化**：约束微调模型偏离预训练参考模型的程度，防止reward hacking并保留生成多样性。

## 可复现要素
- **数据集**：HPDv2（Human Preference Dataset v2，训练集>10万prompt）、Pick-a-Pic v1（公开）；论文明确说明使用这两个数据集。
- **代码**：项目主页 https://fdeng18.github.io/prdp；论文未明确声明GitHub链接，需进一步确认。
- **权重**：基于Stable Diffusion v1.4（公开）进行全量UNet微调。
- **关键超参**：DDPM采样50步、classifier-free guidance 5.0；大规模训练1000 epoch、每epoch 64 prompt × 8图像；小尺度训练100 epoch、每epoch 32 prompt × 16图像；KL正则化强度$\beta=10$（small-scale aesthetic实验）；近端裁剪范围$\epsilon$论文未明确数值。
