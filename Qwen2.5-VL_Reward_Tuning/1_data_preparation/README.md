# 数据与模型准备

本目录包含复现实验所需的数据和模型准备脚本。

## 1. 图片数据集自动化准备脚本 (`prepare_data.sh`)

本脚本专注于下载和整合 `ImageGen-CoT-Reward-5K` 数据集所需的图片文件。它自动化处理了图片的**分片下载、合并、解压及清理**等全部流程。

### 核心特性

*   **全面整合**: 自动处理目录下所有图片数据源（包括`urls.txt`下载的分片图片，以及本地提供的`images_HPD.zip`和`images_OIP.zip`），并将它们的内容全部解压到同一个`images`文件夹中。
*   **高性能下载**: 优先使用 `aria2c` 进行多线程高速下载，若未安装则自动降级为 `wget`，兼具性能与通用性。
*   **鲁棒性设计**: 具备完整的错误处理、断点续传、智能命名和文件校验功能。
*   **自动化流程**: 一站式完成所有图片数据准备工作。

### 使用方法

1.  **准备文件**:
    请确保本脚本与以下文件**位于同一目录**：
    *   `urls.txt` (包含了`images.zip`分片的下载链接，如果使用分片下载方式)
    *   `images_HPD.zip` (如果您本地已存在该文件，脚本会直接解压)
    *   `images_OIP.zip` (如果您本地已存在该文件，脚本会直接解压)

2.  **安装依赖 (推荐)**:
    为了获得最佳下载速度，强烈建议您安装 `aria2c`。脚本还需要 `unzip`。
    ```bash
    # Ubuntu/Debian
    sudo apt-get update && sudo apt-get install aria2 unzip
    # CentOS/Fedora
    sudo yum install aria2 unzip
    # macOS
    brew install aria2 unzip
    ```

3.  **运行脚本**:
    直接在终端中执行：
    ```bash
    bash prepare_data.sh
    ```
    脚本执行完毕后，您将在当前目录下看到一个名为 `images` 的文件夹，其中包含了**所有来源**的图片数据。

## 2. Hugging Face 资源下载脚本 (`download_hf_resources.py`)

本脚本提供了一种自动化、支持镜像源的通用工具，用于从 Hugging Face Hub 下载**模型**和**数据集**。它依赖于 `hfd.sh` 工具来完成高速下载任务。

### 核心特性

*   **模型与数据集下载**: 可配置用于下载 Hugging Face Hub 上的**模型**（如 `Qwen2.5-VL-3B-Instruct`）和**数据集**（如 `CodeGoat24/ImageGen-CoT-Reward-5K`，如果该数据集支持 `hfd.sh` 直接下载）。
*   **自动化工具管理**: 脚本会自动下载 `hfd.sh` 工具并赋予执行权限。
*   **镜像支持**: 自动配置 `hf-mirror.com` 作为 Hugging Face 镜像源，显著提升国内用户下载速度和成功率。
*   **鲁棒性**: 内置错误处理机制，能够捕获和报告下载过程中的问题。
*   **目标路径可配置**: 允许用户方便地指定模型和数据集的本地下载路径。

### 使用方法

1.  **编辑脚本**:
    在使用前，请根据您的需求修改脚本顶部的**配置区**，指定要下载的模型ID、数据集ID（如果需要）以及本地保存路径。
    *   默认情况下，脚本配置为下载 `Qwen/Qwen2.5-VL-3B-Instruct` 模型。
    *   如果需要下载其他模型或**整个Hugging Face数据集**（而非本目录 `prepare_data.sh` 处理的图片文件），请取消脚本中对应配置行的注释。

2.  **运行脚本**:
    在终端中执行：
    ```bash
    python download_hf_resources.py
    ```
    模型和/或数据集将自动下载到脚本中配置的指定路径。
