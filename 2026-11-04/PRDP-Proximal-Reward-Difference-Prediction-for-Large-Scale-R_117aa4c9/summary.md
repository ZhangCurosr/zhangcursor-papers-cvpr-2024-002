---
title: "PRDP-Proximal-Reward-Difference-Prediction-for-Large-Scale-R"
source: https://openaccess.thecvf.com/content/CVPR2024/papers/Deng_PRDP_Proximal_Reward_Difference_Prediction_for_Large-Scale_Reward_Finetuning_of_CVPR_2024_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 00:44:15"
field: "扩散模型对齐与偏好学习"
keywords: ["diffusion model", "reward finetuning", "reinforcement learning", "RLHF", "black-box reward", "proximal policy optimization"]
innovations: ["将RLHF目标转化为等价监督回归的RDP目标，证明完美奖励差异预测即RL最优解", "首次实现10万+提示上稳定的黑盒奖励微调，DDPO在此规模完全失败"]
benchmarks: ["HPDv2", "Pick-a-Pic v1", "HPSv2", "PickScore"]
---

# 论文速读：PRDP-Proximal-Reward-Difference-Prediction-for-Large-Scale-R

## 一句话总结
本文提出了 PRDP（Proximal Reward Difference Prediction），首次实现了在超过 10万条提示的大规模数据集上对扩散模型进行稳定的黑盒奖励微调，通过将其核心推导为等价于强化学习目标的监督回归目标（RDP），避免了策略梯度的高方差问题。

## 研究问题与动机
- **核心问题**：如何在大规模提示数据集上稳定地微调扩散模型以最大化人类偏好奖励。
- **现有方法不足**：现有的 RL-based 方法（如 DDPO、DPOK）依赖策略梯度，在高方差下训练不稳定，无法扩展到大规模提示（DDPO 在超过 400 条提示时即出现崩溃）。
- **KL 正则化难题**：扩散模型的 KL 散度因不可积而无法解析计算，限制了直接应用 RLHF 思路。
- **奖励黑客问题**：缺乏正则化的奖励微调容易触发 reward hacking，导致模型忽略文本对齐而过度优化奖励分数。

## 核心贡献（创新点）
1. **提出 RDP 目标**：将 RLHF 目标转化为等价的监督回归目标，证明完美预测奖励差异的模型即为 RL 目标的最优解。
2. **首次大规模稳定微调**：在超过 10 万条提示的 HPDv2 和 Pick-a-Pic v1 数据集上实现稳定的黑盒奖励微调，而基线方法在此规模下完全失败。
3. **在线优化算法**：引入在线采样策略，每 K 步更新一次快照模型，持续用当前策略分布采样轨迹，显著优于离线数据集方法。
4. **近端更新机制**：借鉴 PPO 的 clipping 思想，对 log 概率比进行逐去噪步骤裁剪，确保训练稳定性。
5. **KL 正则化的显式建模**：通过 RDP 框架自然融入 KL 正则化，有效缓解 reward hacking 并保持图文对齐。

## 方法详解
- **RDP 目标推导**：从 RLHF 目标 $\max_{\pi_\theta} \mathbb{E}[r(\mathbf{x}_0,\mathbf{c}) - \beta \text{KL}[\pi_\theta || \pi_{\text{ref}}]]$ 出发，推导出最优解满足 $\log \frac{\pi_{\theta^*}(\bar{\mathbf{x}}|\mathbf{c})}{\pi_{\text{ref}}(\bar{\mathbf{x}}|\mathbf{c})} = \frac{1}{\beta}r(\mathbf{x}_0,\mathbf{c}) - \log Z(\mathbf{c})$。通过对同一 prompt 的两个轨迹消去 $Z(\mathbf{c})$，得到奖励差异预测的 MSE 损失 $\mathcal{L}(\theta) = \mathbb{E}\|\Delta\hat{r}_\theta - \Delta r/\beta\|^2$。
- **在线优化（Algorithm 1）**：每 epoch 保存模型快照 $\pi_{\theta_{\text{old}}}$，从中采样 denoising trajectories；每 K 次梯度更新后替换快照。
- **近端更新**：借鉴 PPO，对每个去噪步骤 t 的 $\hat{r}_{\theta,t}$ 在 $[\hat{r}_{\theta_{\text{old}},t} - \epsilon, \hat{r}_{\theta_{\text{old}},t} + \epsilon]$ 内 clip，最终 loss 取 unclipped 与 clipped 版本的 max。
- **KL 正则化**：超参数 $\beta$ 控制偏离参考模型的程度，PRDP 天然支持该正则项，而 DDPO 无此设计。

## 实验与结果
- **小尺度实验**（45 条提示，动物名称）：PRDP 在 HPSv2（0.3471）和 PickScore（0.2700）上均略优于 DDPO（0.3398 / 0.2664），验证等效性。
- **大规模实验**（HPDv2 10万+ 提示）：
  - HPSv2：PRDP 在 seen prompts 上得 0.3175，未见过提示（Pick-a-Pic v1 test）0.3050，DDPO 仅 0.2464 / 0.2501 且生成噪声。
  - PickScore：PRDP 0.2424 / 0.2344，DDPO 0.2032 / 0.1992。
- **多奖励联合微调**（Pick-a-Pic v1，权重 PickScore:10, HPSv2:2, Aesthetic:0.05）：PRDP 在复杂 unseen prompts 上生成质量显著优于 SD v1.4 基线。
- **在线 vs 离线**：在线优化在约 10 epoch 即追上离线性能并持续超越。
- **KL 正则化效果**：在 LAION aesthetic 奖励下，DDPO 出现严重 reward hacking（忽略文本提示），PRDP 通过 $\beta=10$ 保持图文对齐。

## 相关工作脉络
- **DDPO [4]**：基于 PPO 的扩散模型 RL 微调，在 400 条提示内稳定，但超过 10万条时崩溃——PRDP 从根本上消除了策略梯度的高方差。
- **DPOK [10]**：引入价值函数估计期望奖励以降低方差，仅在小规模（200 提示）有效——PRDP 完全避免了 RL 框架。
- **DPO [35]**：语言模型中通过转化 RLHF 为目标监督分类任务实现稳定训练；PRDP 受此启发，将其推广到扩散模型并证明等价性。
- **RAFT [8]**：在线重新生成偏好数据并 SFT；PRDP 同样在线但直接优化 RDP 损失，无需 preference 对标注。
- **Diffusion-DPO [50]**：并发工作，同样转化 DPO 到扩散模型，但依赖大规模离线偏好数据；PRDP 支持黑盒奖励且 online 优化。
- **DRaFT [6]**：可微奖励的微调；PRDP 不要求奖励可微，支持任意黑盒奖励模型。

## 局限性与未来方向
- 未探索更复杂的 reward model 架构或多模态奖励。
- 大规模实验仅在文本到图像场景验证，其他模态（视频、3D）的适用性未知。
- 近端更新的超参 $\epsilon$ 和快照更新频率 K 需要调参，缺乏自动搜索策略。
- 未来方向：扩展到条件生成以外的下游任务（如编辑、补全）、探索与 latent diffusion 的更高效结合。

## 研究启发与可借鉴点
1. **RL 到监督学习的转化范式**：将策略梯度目标等价转化为回归/分类损失的思想可迁移到其他生成模型（如 flow-based models）的偏好对齐。
2. **在线快照策略**：每 K 步更新参考分布的策略在训练 GAN 或 VAE 等生成模型时也可能带来稳定性提升。
3. **逐步骤近端裁剪**：将 PPO 的 clip 机制从策略分布扩展到去噪过程的每个时间步，是一个值得推广的细粒度稳定化技巧。
4. **与团队结合的机会**：可将 RDP 框架引入视频生成或多物体组合生成任务，检验其在时序一致性和结构复杂场景下的泛化能力。

## 关键术语表
**PRDP**：Proximal Reward Difference Prediction，本文提出的稳定大规模奖励微调方法。
**RDP**：Reward Difference Prediction，等价于 RLHF 目标的监督回归目标，预测图像对的奖励差异。
**DDPO**：Diffusion-DPO，基于 PPO 的扩散模型强化学习微调方法，本文主要对比基线。
**HPSv2**：Human Preference Score v2，用于评估文本到图像生成质量的奖励模型。
**PickScore**：基于 CLIP 的文本-图像对齐奖励模型，衡量生成图像与提示的相关性。
**Reward hacking**：模型利用奖励函数的缺陷生成高分但低质量的输出。
**Proximal update**：借鉴 PPO 的 clip 机制限制策略更新幅度，防止训练崩溃。
**Online optimization**：用当前模型快照持续采样新轨迹进行训练，而非固定离线数据集。

## 可复现要素
- **数据集**：HPDv2（10万+ 训练提示）、Pick-a-Pic v1（训练+测试集）；HPDv2 训练集可从 DRaFT [6] 获取，Pick-a-Pic v1 为公开数据集。
- **代码**：论文提供了 project page https://fdeng18.github.io/prdp，但正文未明确声明 GitHub 链接；权重未开源。
- **关键超参**：$\beta=10$（KL 正则强度）、50 步 DDPM sampler、guidance scale=5.0、epoch=1000（大规模）、每 epoch 64 prompts×8 images、K=512 次梯度更新后快照更新。
