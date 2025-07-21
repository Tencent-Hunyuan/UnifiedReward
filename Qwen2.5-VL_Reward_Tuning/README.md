# Qwen2.5-VL-3B 奖励模型微调实验复现指南

本项目由MuGuang-L完成，作为腾讯犀牛鸟计划 `UnifiedReward` 项目的一部分。
本目录包含了复现我的《全参与LoRA对比分析实验报告》所需的所有代码、配置文件和详细说明。

## 目录结构

```markdown
├── EXPERIMENT_REPORT.md # 详细的图文实验报告，包含所有结论与分析
├── loss_figure/ # 存放训练Loss和验证Loss的静态图表
├── evaluation/ # 新增！存放lora模型合并脚本与评测结果
├── tensorboard_logs/ # 存放原始 TensorBoard 日志文件，支持本地交互式查看
├── 1_data_preparation/ # 存放所有数据和模型准备相关的脚本
├── 2_training_scripts/ # 存放全参和LoRA的训练配置文件
└── README.md # 您正在阅读的这份总说明文件
```
## 复现流程

**第一步：准备数据与模型**

所有数据和模型准备相关的脚本和详细说明位于 `1_data_preparation/` 目录下。请先进入该目录并按其说明操作，确保所有必要文件（如图片数据集和Qwen基座模型）都已准备就绪。

**第二步：开始训练**

`SFT`和`LoRA`的训练配置文件位于 `2_training_scripts/` 目录下。请在使用前，确保根据您的本地环境修改`.yaml`文件中的模型和数据路径。注意，所有的训练均在 llamafactory 框架下完成：

*   **运行SFT训练 (示例)**:
    ```bash
    llamafactory-cli train 2_training_scripts/qwen_3b_sft.yaml
    ```
*   **运行LoRA训练 (示例)**:
    ```bash
    llamafactory-cli train 2_training_scripts/qwen_3b_lora.yaml
    ```

**第三步：模型合并与性能评测**

训练完成后，对于LoRA微调，我们需要将LoRA权重合并到基座模型中，并使用评测脚本验证：在不同微调方法下模型的性能变化。

1.  **合并LoRA权重**(可选，全参微调则跳过该步骤):
    使用 `evaluation` 目录下的配置文件，将训练好的LoRA权重合并到原始模型中。请确保已根据您的环境修改 `qwen_2.5vl_3b_lora_merge.yaml` 文件中的模型、适配器和导出路径。
    ```bash
    llamafactory-cli export evaluation/qwen_2.5vl_3b_lora_merge.yaml
    ```

2.  **运行性能评测**:
    模型合并后，我们使用 `UnifiedReward` 项目提供的评测脚本，在 `GenAI-Bench` 上验证模型的性能（这里的路径仅用于举例）。
    
    ```bash
    python /work/UnifiedReward/benchmark_evaluation/GenAI-Bench-Image/qwen_genAI_bench_image_test.py
    ```
    评测完成后，会生成性能图表，你可以在 `EXPERIMENT_REPORT.md` 中查看对评测结果的详细分析。

**第四步：查看详细实验报告与训练日志**

*   **实验报告**: 所有实验的详细结果、图表和深度分析，请查阅根目录下的 `EXPERIMENT_REPORT.md` 文件。
*   **交互式训练日志**: 如果您想交互式地查看训练Loss、验证Loss等详细指标，可以参考 `tensorboard_logs/` 目录下的 `README.md` 文件，在本地启动 TensorBoard。

