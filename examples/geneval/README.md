# Geneval 使用说明

本仓库基于 [Geneval](https://github.com/djghosh13/geneval)，用于文生图模型（如 Stable Diffusion 3.5）在 **Geneval 指标**下的评测。本文档说明了环境安装、模型下载、自定义脚本及完整评测流程。

---

## 环境准备与依赖安装

1. **克隆仓库与下载模型**

    ```bash
    git clone https://github.com/djghosh13/geneval.git
    cd geneval

    # 下载目标检测器 ckpt（会存放在 <OBJECT_DETECTOR_FOLDER>/ 下）
    ./evaluation/download_models.sh "<OBJECT_DETECTOR_FOLDER>/"
    ```

2. **安装依赖（推荐 Python 3.10+，CUDA 12.1）**

    ```bash
    # PyTorch 与相关依赖
    pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

    # 基础依赖
    pip install open-clip-torch==2.26.1
    pip install clip-benchmark
    pip install -U openmim
    pip install einops
    python -m pip install lightning
    pip install 'diffusers[torch]' transformers
    pip install tomli
    pip install platformdirs
    pip install --upgrade setuptools   # 确保 setuptools 最新
    ```

3. **安装 mmcv 与 mmdet（需在 pip 安装完成后执行）**

    ```bash
    # 安装 mmcv
    git clone https://github.com/open-mmlab/mmcv.git
    cd mmcv
    git checkout 1.x
    MMCV_WITH_OPS=1 MMCV_CUDA_ARGS="-arch=sm_90" pip install -v -e .

    # 确保 nvcc 对应 CUDA 12.1
    conda install cuda-nvcc -c nvidia/label/cuda-12.1.0

    # 安装 mmdet
    git clone https://github.com/open-mmlab/mmdetection.git
    cd mmdetection
    git checkout 2.x
    pip install -v -e .
    ```

---

## 自定义脚本说明

请将本文件夹中的脚本放置到 `geneval/generation` 路径下，并根据实际路径修改。

- `diffusers_generate-sd.py`  
  使用 **Stable Diffusion 3.5-Medium** 直接生成图片。

- `examples/geneval/diffusers_generate-sd3.5-lora.py`  
  使用 **Stable Diffusion 3.5-Medium + DiffusionDPO LoRA SFT** 生成图片。

---

## 完整评测流程

我们提供了两个一键运行脚本：

### 1. Baseline（SD3.5-Medium）

运行以下命令：

```bash
bash baseline.sh

### 2. LoRA SFT 优化版（SD3.5-Medium + DiffusionDPO）

运行以下命令：

```bash
bash sd3.5-lora.sh
