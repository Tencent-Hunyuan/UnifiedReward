# **调研报告：文本生成图像强化学习框架**
## 1.摘要
文本生成图像技术是当前多模态生成领域的研究热点，而强化学习为解决T2I任务中的语义对齐和生成可控性问题提供了新的思路。
本报告首先回顾了传统的文本生成图像算法，随后介绍了两个经典的RL-T2I框架——DDPO和ReFL，这两个框架奠定了基于RL的T2I方法的基础。另外，本报告深入分析了两个较为前沿的框架：T2I-R1和GoT-R1，它们分别引入了双层CoT推理和推理链优化策略，实现了对复杂提示的更好处理以及多对象生成任务的性能突破。本报告通过对现有文献的综合分析，提供了关于RL在T2I中应用的全面视图，并展望了未来可能的发展方向和技术挑战。




## 2.文本生成图像方法概览


文本生成图像（Text-to-Image, T2I）技术的发展在过去几年中取得了显著的进步，这主要得益于深度学习模型的不断演进和优化。当前，T2I领域的主要的基础网络架构包括基于GAN的文本生成图像方法普遍采用的GAN模型、条件生成对抗网络
（conditional generative adversarial network，CGAN）
和自编码器架构、基于自回归模型架构的方法所采用的AR模型架构和对比模型（contrastive language-image pre-training， CLIP）、基于扩散模型架构生成图像方法所采用的扩散模型架构.。这些模型各自具有独特的优势，并在不同的应用场景中展现出卓越的表现。

### 1. GAN
- 定义：通过生成器（G）与判别器（D）的对抗训练生成图像，优化目标为最小化生成分布与真实分布的差异。
- 核心技巧：

    生成器损失 $L_{G}^{\text{uncon}} = -E_{x\sim P_G}[\log_2(D(x))]$

    判别器损失 $L_{D}^{\text{uncon}} = -\frac{1}{2}E_{x\sim P_{\text{data}}}[\log_2(D(x))] - \frac{1}{2}E_{x\sim P_G}[\log_2(1-D(x))]$

- 代表工作：Generative Adversarial Text to Image Synthesis（Reed et al., 2016）[1]
- 优势／局限：

    优势：模型轻量，训练速度快。

    局限：模式崩溃问题严重，生成图像多样性差；文本控制需额外条件输入。

### 2. CGAN
- 定义：在GAN框架中引入文本描述 $\phi$ 作为条件输入，实现文本到图像的映射。
- 核心技巧(改进点)：

    条件损失函数 $L_{G}^{\text{con}} = -E_{x\sim P_G}[\log_2(D(x,\phi))]$

    文本特征通过LSTM(早期)和AE自编码器提取
- 代表工作：Conditional Generative Adversarial Nets（Mehdi Mirza and Simon Osindero, 2014）[2]
- 优势／局限：

    优势：支持细粒度文本控制（如词级注意力）。

    局限：需多层次嵌套架构（如StackGAN++）提升分辨率，复杂度高。
### 3. 对比模型
- 定义：通过大规模数据训练，使模型能够理解文本描述与图像内容之间的语义关联
- 核心技巧：

    对比学习

    大规模数据预训练

    双编码器架构

    零样本分类


- 代表工作：Zero-shot text-to
image generation（Ramesh et al., 2021）[3]
- 优势／局限：

    优势：强大的跨模态对齐能力（如无需人工标注的类别标签）。

    局限：模态不平衡、细粒度理解不足、依赖自然语言描述。

### 4.自回归模型（AR）
- 定义：以序列生成方式逐像素/块预测图像，依赖Transformer或RNN建模长程依赖。
- 核心技巧：

    联合训练CLIP模型对齐文本-图像特征

    两阶段生成
- 代表工作：Scaling Autoregressive Models for Content-Rich Text-to-Image Generation（Jiahui Yu et al., 2021）[4]
- 优势／局限：

    优势：生成图像连贯性强。

    局限：参数量大（如Parti达200亿），推理速度慢。

### 5. 扩散模型（Diffusion）
- 定义：通过前向加噪与逆向去噪过程生成图像，隐空间建模提升效率。
- 核心技巧：

    分类器引导

    潜在扩散
- 代表工作：Re-Imagen: Retrieval-Augmented Text-to-Image Generator（Wenhu Chen et al., 2022）[5]
- 优势／局限：

    优势：SOTA图像质量。

    局限：需工业级算力训练，采样步骤多（通常>100步）。

然而，尽管上述方法在图像生成方面取得了巨大成功，它们在语义对齐、生成可控性以及满足人类偏好等方面仍存在不足。强化学习（Reinforcement Learning, RL）作为一种优化策略，能够在一定程度上弥补这些缺陷。通过引入奖励机制，RL可以引导模型生成更加符合用户意图和偏好的图像，从而提升整体生成效果。特别是在处理复杂提示或多对象生成任务时，RL的应用显得尤为重要。

## 3.文本生成图像强化学习方法流程概览
强化学习（RL）通过精心设计的奖励模型能够为系统提供更加丰富且具有指导性的反馈信号，从而有效缓解奖励稀疏性这一关键问题。在此基础上，采用PPO、GRPO等先进的强化学习算法不仅可以显著提升训练过程的稳定性，还能更合理地协调探索与利用之间的动态平衡。更进一步地，通过整合语义对齐、美学质量和多样性等多维度的奖励机制，该系统能够大幅提升生成图像在语义一致性方面的表现，同时显著改善其整体视觉质量。

::: mermaid
flowchart TB
  subgraph 文本生成阶段
    A[文本Prompt输入] --> B[Stable Diffusion多候选生成]
    B --> B1[生成N张候选图像]
  end

  subgraph 多维度评估
    B1 --> C[UnifiedReward评估模块]
    C --> C1[CLIP语义相似度]
    C --> C2[美学评分（如AesBench）]
    C --> C3[空间逻辑评分（如GLIGEN）]
  end

  subgraph 强化学习阶段
    C1 & C2 & C3 --> D[加权综合得分排序]
    D --> E[选择Top-K高分样本]
    E --> F[PPO策略梯度更新]
    F -->|反馈循环| B
  end

  F --> G[最终优化图像输出]
:::

## 4.文本生成图像强化学习方法概览


本调研旨在深入分析现有的文本生成图像（T2I）强化学习框架，并探索奖励模型的优化潜力。我们将重点研究四个具有代表性的框架：DDPO（Denoising Diffusion Policy Optimization）、ReFL（Reward Feedback Learning）、DanceGRPO 以及 T2I-R1（双层CoT+RL）。通过对这些框架的详细分析和相互对比，我们希望能够为未来的研究提供有价值的见解，并提出改进建议。

### 1. ReFL（Reward Feedback Learning for Diffusion Models）
- 定义：直接利用ImageReward的反馈优化扩散模型，通过梯度反向传播调整生成过程。
- 核心技巧：
    在去噪步骤的后半段（30-40步）引入RM分数作为损失信号。

    结合预训练损失与奖励损失稳定训练。

    $\mathcal{L}_{pre} = \mathbb{E}_{(y_i, x_i) \sim \mathcal{D}} \left[ \mathbb{E}_{\mathcal{E}(x_i), y_i, \epsilon \sim \mathcal{N}(0,1), t} \left[ \| \epsilon - \epsilon_\theta(z_t, t, \tau_\theta(y_i)) \|^2_2 \right] \right]$

    $ \mathcal{L}_{reward} = \lambda \mathbb{E}_{y_i \sim \mathcal{Y}} \left[ \phi(r(y_i, g_\theta(y_i))) \right]$

    随机选择去噪步骤 $t$ 以避免过拟合。
- 奖励函数实验结果

    * 评估指标：

        人类偏好排名：在466条真实用户提示上，ReFL生成图像的胜率58.79%。
        | Methods           | Real User Prompts WinRate | MT Bench WinRate |
        |-------------------|---------------------------|------------------|
        | SD v1.4 (baseline) | -                         | -                |
        | Dataset Filtering   | 55.17                     | 51.72            |
        | Reward Weighted     | 39.52                     | 43.33            |
        | RAFT  (iter=1)       | 49.86                     | 42.31            |
        | RAFT (iter=2)            | 30.85                     | 33.02            |
        | RAFT (iter=3)            | 20.97                     | 26.19            |
        | ReFL (Ours)              | 58.79                     | 58.49            |

        自动指标：ImageReward分数提升显著
        | Model             | Preference Acc. |
        |-------------------|-----------------|
        | CLIP Score        | 54.82           |
        | Aesthetic Score   | 57.35           |
        | BLIP Score        | 57.76           |
        | ImageReward (Ours)| 65.14           |

- 代表工作/论文：ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation（Jiazheng Xu et al., 2023）[6]。
- 优势／局限：

    优势：直接优化生成模型，优于间接方法（如数据过滤、损失加权）；人类评估胜率显著高于基线（58.79%）。

    局限：
依赖RM的准确性；仅适用于扩散模型，无法直接用于其他生成架构（如GAN）。


### 2. DDPO（Denoising Diffusion Policy Optimization）

- 定义：将去噪过程建模为多步决策问题，利用策略梯度算法直接优化扩散模型的下游目标（而非近似似然）。
- 核心技巧：

    将去噪步骤映射为马尔可夫决策过程（MDP），通过策略梯度算法在每个去噪步骤中进行优化。

    支持两种梯度估计器：Score Function（DDPO<sub>SF</sub>）和Importance Sampling（DDPO<sub>IS</sub>）。

    CFG训练机制稳定引导过程
- 奖励函数系统
    | 奖励类型     | 计算方式                           | 优化目标           | 参数设置       |
    |---------------------|------------------------------------|--------------------|----------------------------|
    | 图像压缩性   | -log(JPEG文件大小)                 | 最小化存储空间     | 质量因子Q=75   |
    | 美学评分     | LAION预测器(CLIP线性层)            | 最大化人类审美评分 | 输入分辨率512x512 |
    | 文本对齐度   | BERTScore(LLaVA描述,原始提示)      | 提升语义匹配度     | 温度参数τ=0.01 |

- LLaVA 反馈（自动化提示对齐）
    * 定义：利用视觉语言模型（VLM，如LLaVA）自动生成图像描述，通过语义相似度（如BERTScore）评估与提示的一致性。
     * 核心技巧：

        将VLM描述与原始提示的相似度作为奖励信号。

        支持复杂目标（如“动物进行非常规活动”）的优化。

    * 核心架构：

        视觉编码器：CLIP-ViT（提取图像特征）

        语言模型：Vicuna（LLaMA 的指令微调版本）

        连接方式：通过线性投影将图像特征映射到语言模型的嵌入空间。
        | 方法             | 依赖人工标注？ | 可扩展性 | 适用任务范围         |
        |------------------|----------------|----------|----------------------|
        | 传统人类反馈（RLHF） | 是            | 低       | 简单、明确的目标     |
        | VLM 自动奖励     | 否            | 高       | 复杂、抽象的目标     |
    * 优势／局限：

        优势：避免人工标注成本，适用于难以编程定义的奖励（如语义对齐）。

        局限：依赖VLM的识别能力，可能被对抗样本攻击（如文本拼写错误）。

- 实现细节

    奖励归一化

    多步奖励分配

    避免奖励过优化(早停和KL正则化)

- 代表工作/论文：Training Diffusion Models with Reinforcement Learning（Black et al., 2024）[7]。

- 奖励函数对比实验结果：
    | 方法         | 压缩性任务 | 美观性任务 | 文本对齐任务 |
    |--------------|------------|------------|--------------|
    | RWR          | 中等       | 低         | 低           |
    | DDPO (SF)    | 高         | 中         | 中           |
    | DDPO (IS)    | 最高       | 高         | 高           |

- 优势／局限：

    优势：直接优化下游目标（如图像美观性、压缩率），而非近似似然。

    在多任务（如文本-图像对齐）中表现优于RWR（Reward-Weighted Regression）。

    局限：需要设计合适的奖励函数；可能因奖励过优化导致模型偏离原始分布。

### 3. DanceGRPO
- 定义：首个将Group Relative Policy Optimization（GRPO）应用于视觉生成任务（扩散模型与整流流）的统一强化学习框架，支持文本到图像、文本到视频、图像到视频等多种任务。
- 核心技巧：

    将扩散模型和整流流的采样过程重新表述为马尔可夫决策过程（MDP）。

    通过随机微分方程（SDE）统一采样过程，引入GRPO目标函数优化策略。

    采用共享初始化噪声和动态时间步选择策略提升训练稳定性。
- 技巧细节
    * 多奖励模型集成（报告的关注点）

        定义：结合图像/视频美学、文本对齐、运动质量等多种奖励信号优化生成策略。

        核心技巧：

            基于优势函数聚合不同尺度的奖励（而非直接加权求和）。

            通过CLIP分数等辅助奖励缓解单一奖励过优化（如“油腻”图像）。
        | 奖励模型               | 任务适用性                         | 优化目标                                       | 模型来源         |
        |------------------------|------------------------------------|------------------------------------------------|------------------|
        | Image Aesthetics       | 文本→图像（T2I）                   | 图像美学质量（如构图、色彩、细节）             | HPS-v2.1      |
        | Text-Image Alignment   | 文本→图像（T2I）                   | 文本与图像的语义一致性                         | CLIP Score    |
        | Video Aesthetics Quality | 文本→视频（T2V）、图像→视频（I2V） | 视频帧的美学质量（清晰度、光影等）             | VideoAlign    |
        | Video Motion Quality   | 文本→视频（T2V）、图像→视频（I2V） | 视频运动的自然度（如物理合理性、流畅性）       | VideoAlign    |
        | Thresholding Binary Reward | 所有任务                     | 二元奖励（0/1），用于强化关键优化目标           | 基于 HPS/CLIP 阈值化 |

       * HPS-v2.1（Human Preference Score）是一个基于人类偏好的图像美学评估模型。

        * CLIP Score 衡量文本和图像的语义匹配度。

        * VideoAlign 提供视频美学（VQ）和运动质量（MQ）两个维度的评估。

        * Binary Reward 将连续奖励（如 HPS > 0.28）转换为 0/1 信号，用于稀疏反馈学习。（二元奖励学习）
        优势／局限：

        优势：生成结果更自然，避免奖励黑客现象。

        局限：需设计奖励权重，部分奖励模型（如VideoAlign文本对齐维度）可能不稳定。

    * Best-of-N推理扩展

    *  时间步选择策略
- 实验结果
    | Method | RL-based | VideoGen | Scalability | Reward ↑ | RFs | No Diff-Reward |
    |--------|----------|----------|-------------|----------|-----|----------------|
    | DDPO/DPOK | ✓        | ✗        | ✗           | ✓        | ✗   | ✓              |
    | ReFL    | ✗        | ✗        | ✓           | ✓        | ✓   | ✗              |
    | DPO     | ✗        | ✓        | ✓           | ✗        | ✓   | ✓              |
    | Ours    | ✓        | ✓        | ✓           | ✓        | ✓   | ✓              |

        结论：DanceGRPO是目前唯一同时支持 多奖励、不可微奖励、视频生成的RLHF方法。    

- 代表工作/论文：

    DanceGRPO: Unleashing GRPO on Visual Generation（Xue et al., 2025）[8]。
- 优势／局限：

    优势：

    跨任务、模型和奖励模型的统一适配性。

    在HPS-v2.1、CLIP Score等基准上最高提升181%性能。

    支持稀疏二元奖励学习和Best-of-N推理扩展。

    局限：

    依赖高质量奖励模型，视频生成任务计算成本高。

    需调整噪声水平和时间步选择策略以平衡效率与效果。



### 4. T2I-R1:Semantic-level CoT（双层CoT+RL）

- 定义：在图像生成前进行高级语义规划，通过文本推理生成图像的全局结构（如物体外观、位置等）。

- 核心技巧：将图像生成分解为语义规划和像素生成两个阶段，利用文本推理优化生成过程。

- 技巧细节：
    * Token-level CoT

        * 定义：将图像生成过程视为逐块（patch-by-patch）的视觉推理链，通过局部像素生成逐步构建完整图像。

        * 核心技巧：将图像生成类比为文本推理链，优化中间生成步骤以提升细节一致性和视觉连贯性。

        * 优势／局限：

            优势：提升局部细节质量和跨区域视觉一致性；适用于低层次像素优化。

            局限：单独使用可能降低图像多样性（需结合语义级CoT）。

    * BiCoT-GRPO（协同优化双层次CoT）
        * 定义：通过强化学习（GRPO）联合优化语义级和token级CoT，在同一训练步骤中协调全局规划与局部生成。
        * 核心技巧：

            语义级CoT生成文本规划。

            Token级CoT生成图像块，通过多专家奖励模型评估生成质量。

        * 优势／局限：

            优势：显著提升生成性能（T2I-CompBench +13%，WISE +19%）；支持复杂提示和罕见场景。

            局限：依赖多奖励模型集成，训练复杂度较高。

    | 方法           | 核心目标       | 依赖能力         | 典型提升场景               |
    |----------------|----------------|------------------|----------------------------|
    | Semantic-level | 高级语义规划   | 文本推理         | 复杂关系、用户意图理解     |
    | Token-level    | 局部细节生成   | 像素级连贯性     | 细节质量、视觉一致性       |
    | BiCoT-GRPO     | 双层次协同优化 | 多模态奖励模型   | 综合性能（SOTA基准）       |
- 实验结果（部分）：
    * T2I-CompBench Performance

    | Model        | Color (↑) | Shape (↑) | Texture (↑) | Spatial (↑) | Non-Spatial (↑) | Complex (↑) |
    |--------------|-----------|-----------|-------------|-------------|-----------------|-------------|
    | Janus-Pro-7B | 0.6359    | 0.3528    | 0.4936      | 0.2061      | 0.3085          | 0.3559      |
    | FLUX.1       | 0.7407    | 0.5718    | 0.6922      | 0.2863      | 0.3127          | 0.3703      |
    | T2I-R1 (Ours)| 0.8130    | 0.5852    | 0.7243      | 0.3378      | 0.3090          | 0.3993      |

        结论：在T2I-CompBench上相比基线提升13%，在WISE基准上提升19%；在T2I-CompBench的5/6个子任务上超越当前最先进的扩散模型FLUX.1
    * Ablation Study Results（单个成分影响）

    | Configuration         | Color (↑) | Shape (↑) | Texture (↑) | Diversity (↑) |
    |-----------------------|-----------|-----------|-------------|---------------|
    | Baseline (Janus-Pro)  | 0.6359    | 0.3528    | 0.4936      | 6.976         |
    | + Semantic-only CoT   | 0.8082    | 0.5684    | 0.7219      | 8.177         |
    | + Token-only CoT      | 0.7752    | 0.5849    | 0.7451      | 6.255         |
    | Full T2I-R1           | 0.8130    | 0.5852    | 0.7243      | 8.203         |
        Token-level CoT的贡献：纹理生成质量提升（0.4936→0.7243）；空间关系准确率提升（0.2061→0.3378）；单独使用时虽提高形状准确率但会降低多样性（6.255 vs 8.203）
- 奖励函数设计（报告关注点）：
    * 多专家奖励模型组合

        | 奖励类型       | 具体模型     | 计算方式                                       | 优化目标   | 权重 |
        |----------------|--------------|------------------------------------------------|------------|------|
        | 人类偏好 (HPM) | HPS v2       | 生成图像与人类审美偏好的匹配度                 | 整体美观性 | 0.4  |
        | 物体检测       | GroundingDINO| 物体存在性(1/0) + 空间关系得分 (0-1)           | 对象准确性 | 0.3  |
        | 视觉问答 (VQA) | GIT-Large    | 属性验证正确率 (P_yes/(P_yes+P_no))             | 细节匹配度 | 0.2  |
        | 对齐评估 (ORM) | LLaVA-7B     | 整体提示-图像对齐度(0-1)                       | 语义一致性 | 0.1  |
    * 奖励组合策略

        | 策略     | 公式                                                                 | 特点           |
        |----------|----------------------------------------------------------------------|----------------|
        | 加权平均 | $R_{total} = 0.4*R_{HPM} + 0.3*R_{Det} + 0.2*R_{VQA} + 0.1*R_{ORM}$               | 默认方案       |
        | 动态调整 | $w_i = softmax(R_i/τ), τ=0.1$                                          | 根据各奖励方差自动调节 |
        | 阈值过滤 | $R_{total} = ΣR_i * I(R_i>θ), θ=0.5$                                      | 过滤低质量生成 |

        | 奖励组合 | T2I-CompBench | WISE     | 训练稳定性 |
        |----------|---------------|----------|------------|
        | 仅HPM    | 0.72±0.15     | 0.48±0.12| 容易过拟合 |
        | HPM+Det  | 0.79±0.08     | 0.52±0.07| 较稳定     |
        | 全组合   | 0.81±0.05     | 0.54±0.04| 最稳定     |
            
            结论：通过多专家奖励的相互制约，防止模型"欺骗"单一奖励指标（如生成虚假高置信度检测框但实际不可见的物体）

- 代表工作/论文：T2I-R1: Reinforcing Image Generation with Collaborative Semantic-level and Token-level CoT（Jiang et al., 2025）[9]。
- 优势／局限：

    优势：显式分离推理与生成，提升复杂场景和罕见情况的生成质量；增强模型对用户意图的理解。

    局限：依赖模型的文本推理能力，可能增加计算复杂度。


## 5.总结与展望
#### 现状分析
当前T2I生成框架各具优势，但存在生成质量、语义一致性与计算效率的平衡问题。强化学习在T2I任务中通过状态（生成质量）、动作（参数调整）和奖励（质量评估）三要素优化生成过程，应用场景包括：

* 图像质量提升：通过FID等指标设计奖励函数

* 多模态融合：优化文本与其他模态信息的结合

* 动态生成优化：增强实时生成任务的适应性

#### 核心挑战

* 奖励函数设计复杂（需平衡多维度目标）

* 训练成本高且稳定性不足

#### 未来可能方向

- 技术改进

    * 采用离线强化学习降低交互成本

    * 探索多智能体协作分工（如文本理解与图像生成分离）

    * 开发动态奖励调整机制（自适应权重策略）

- 奖励模型优化

    * 结合人类偏好数据提升语义对齐

    * 采用UnifiedReward-Think 等一体化多模态思维链（CoT）推理工具，结合视觉-语言对齐优化和分层语义推理，增强模型的多模态理解与生成能力

总结：
随着算法改进与硬件升级，强化学习有望推动T2I系统向更高效、高质、可控的方向发展，尤其在动态场景和跨模态生成领域潜力显著。


## 参考文献

[1] REED S, AKATA Z, YAN X, et al. Generative adversarial text to image synthesis [C]//International Conference on Machine Learning. New York: ACM, 2016: 1060-1069.

[2] Mirza, M., & Osindero, S.(2014). Conditional Generative Adversarial Nets (No. arXiv:1411.1784). arXiv. https://doi.org/10.48550/arXiv.1411.1784


[3] RAMESH A, PAVLOV M, GOH G, et al. Zero-shot text-to
image generation [C]//International Conference on Machine
 Learning. [S. l. ]: ACM, 2021: 8821-8831.

[4] Yu, J., Xu, Y., Koh, J. Y., et al(2022). Scaling Autoregressive Models for Content-Rich Text-to-Image Generation (No. arXiv:2206.10789). arXiv. https://doi.org/10.48550/arXiv.2206.10789

[5] Chen, W., Hu, H., Saharia, C., & Cohen, W. W. (2022). Re-Imagen: Retrieval-Augmented Text-to-Image Generator (No. arXiv:2209.14491). arXiv. https://doi.org/10.48550/arXiv.2209.14491

[6].Xu, J., Liu, X., Wu, Y., Tong, Y., Li, Q., Ding, M., Tang, J., & Dong, Y. (2023). ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation (No. arXiv:2304.05977). arXiv. https://doi.org/10.48550/arXiv.2304.05977

[7].Black, K., Janner, M., Du, Y., Kostrikov, I., & Levine, S. (2024). Training Diffusion Models with Reinforcement Learning (No. arXiv:2305.13301). arXiv. https://doi.org/10.48550/arXiv.2305.13301

[8].Xue, Z., Wu, J., Gao, Y., Kong, F., Zhu, L., Chen, M., Liu, Z., Liu, W., Guo, Q., Huang, W., & Luo, P. (2025). DanceGRPO: Unleashing GRPO on Visual Generation (No. arXiv:2505.07818). arXiv. https://doi.org/10.48550/arXiv.2505.07818

[9].Jiang, D., Guo, Z., Zhang, R., Zong, Z., Li, H., Zhuo, L., Yan, S., Heng, P.-A., & Li, H. (2025). T2I-R1: Reinforcing Image Generation with Collaborative Semantic-level and Token-level CoT (No. arXiv:2505.00703). arXiv. https://doi.org/10.48550/arXiv.2505.00703







