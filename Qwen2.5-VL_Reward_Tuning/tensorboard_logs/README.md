# TensorBoard 训练日志查看指南

本目录包含本次实验中全参微调和 LoRA 微调训练过程的原始 TensorBoard 日志文件。

您可以按照以下步骤在本地启动 TensorBoard 应用，交互式地查看训练指标（如 Loss 曲线）。

### 步骤

1.  **安装 TensorBoard**:
    如果您尚未安装 TensorBoard，请通过 pip 进行安装：
    ```bash
    pip install tensorboard
    ```

2.  **启动 TensorBoard**:
    在当前目录 (即包含 `SFT_full_tuning` 和 `LoRA_tuning` 文件夹的 `tensorboard_logs/` 目录) 下，运行以下命令：
    ```bash
    tensorboard --logdir .
    ```
    或者，如果您在项目根目录下运行，命令可能是：
    ```bash
    tensorboard --logdir ./tensorboard_logs
    ```
    （请根据您实际提交时 `tensorboard_logs` 文件夹的相对路径进行调整）

3.  **访问 TensorBoard 界面**:
    命令运行后，终端会显示一个本地访问地址（通常是 `http://localhost:6006`）。在浏览器中打开此地址，即可看到交互式的 TensorBoard 界面。

4.  **查看对比曲线**:
    在 TensorBoard 界面的左侧，您可以看到不同的“Run”。勾选您想要对比的 Runs，它们的 Loss 曲线（train/loss 和 eval/loss）将自动显示在同一张图表中，方便您进行详细对比分析。