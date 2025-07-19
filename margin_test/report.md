# 探索测试集中pair对的优劣差距对模型评测的影响

**完成人：Met4physics**

## 实验环境：

| 项目 |        参数         |
| :----------: | :----------------------------------: |
| GPU | Nvidia RTX 4090 24G |
| CUDA | 12.1                      |
| Transformers | 4.49.0 |
| Torch | 2.1.2 |

## GenAI-Bench Image Generation

### 实验结果

当指示模型直接生成Pair对的好坏时，UnifiedReward-Qwen 7b在GenAI-Bench Image Generation上的表现：

| Metrics     | Acc. |
| :----------: | :----------------------------------: |
| w/ tie | 57.11                           |
| w/o tie | 76.76                           |

使用的Prompt为：
```
You are given a text caption and two generated images based on that caption. Your task is to evaluate and compare these images based on two key criteria:\n1. Alignment with the Caption: Assess how well each image aligns with the provided caption. Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n2. Overall Image Quality: Examine the visual quality of each image, including clarity, detail preservation, color accuracy, and overall aesthetic appeal.\nCompare both images using the above criteria and select the one that better aligns with the caption while exhibiting superior visual quality.\nProvide a clear conclusion such as \"Image 1 is better.\", \"Image 2 is better.\" and \"Both images are equally good.\"\nYour task is provided as follows:\nText Caption: [{prompt}]
```

可以看到与最初论文中使用的LLaVA-OneVision-7B相比，Qwen作为基座模型有着更好的性能。

当指示模型生成Pair两张图片的评分时，UnifiedReward-Qwen 7b在GenAI-Bench Image Generation上的表现：

| Metrics     | Acc. |
| ------------ | ------------------------------------ |
| w/ tie | 34.56                           |
| w/o tie | 46.44                           |

使用的Prompt为：
```
You are given a text caption and a generated image based on that caption. Your task is to evaluate this image based on two key criteria:\n1. Alignment with the Caption: Assess how well this image aligns with the provided caption. Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n2. Overall Image Quality: Examine the visual quality of this image, including clarity, detail preservation, color accuracy, and overall aesthetic appeal.\nFrom 0 to 100, how much do you rate for this image in terms of the overall image quality and alignment with the text caption?\nDo not dominant the rating by a single attribute such as image quality, but a overall rating on the above 2 factors.\nProvide a few lines for explanation and the rate number at last after \"Final Score:\".\nYour task is provided as follows:\nText Caption: [{prompt}]
```

具体的运行代码可以参考`margin_test/GenAI_eval.py`

### 结果分析

![](E:\UnifiedReward\margin_test\margin_vs_accuracy.png)

如图所示，我们将Margin以10为单位进行分组，可以看到，随着Margin增大，准确率也在上升。由于模型输出具有不稳定性，有时无法提取出Final Score，我们遇见此类情况时将Final Score直接置0，我们排除所有包含0的数据后的分布图为：

![](E:\UnifiedReward\margin_test\margin_vs_accuracy_filter_zero.png)

可以看到，`(90, 100]`区间表现的并不正常。此区间共有3个样本，他们的评分分别为`95,1`、 `95,1`和 `1,95`，均为一个95和一个1。因此，奖励模型可能在这三个样本上出现了不稳定的情况。整体来讲，随着Margin增大，准确率也在增大。

## 总结

实验结果较好的符合了我们的预期，即“Margin越大，效果越好”。但是仍然存在一些问题：

1. 奖励模型的输出并不稳定，会出现不按照Prompt要求输出分数的情况。
2. 奖励模型输出的分数并不精确，99.62%的分数都是5的倍数，说明奖励模型本质上还是以0.05为颗粒度进行打分。
3. 奖励模型在特定样本上会出现误打分的情况。

针对所提出的问题，有以下几个方面值得我们研究：

1. 如何稳定奖励模型的输出。有两个方向值得考虑：
   1. 设计更好的Prompt来进行对齐。（不能保证100%）
   2. 给模型加上一个回归头，专门用来输出分数。（会出现回归结果和文字输出结果不相等的情况）
2. 对于相近图像，收集更好的数据进行进一步SFT，能够改善低Margin时的性能。
3. 奖励模型输出分数的精确性，可以由设计更好的Prompt来满足。