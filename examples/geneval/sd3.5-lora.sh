#!/usr/bin/env bash
set -euo pipefail

export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
# 更稳的 allocator 组合：加上垃圾回收阈值，显式限制分块阈值
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:128"
# 显式绑定 GPU 顺序，避免 rank 误上 0 号卡
export CUDA_VISIBLE_DEVICES=0,1
# 让 accelerate 明确知道绑哪两张卡
ACC_GPU_IDS="0,1"

PROMPT_DIR="geneval/prompts"
IMAGE_FOLDER="sd3.5_large/baseline/IMAGE_FOLDER_sd3.5_lora"
RESULTS_FOLDER="sd3.5_large/baseline/RESULTS_FOLDER"
OBJECT_DETECTOR_FOLDER="geneval/<OBJECT_DETECTOR_FOLDER>"

# —— 1. 图像生成 —— #
# accelerate launch \
#   --multi_gpu --num_processes=2 --gpu_ids="0,1" --mixed_precision=fp16 \
#   geneval/generation/diffusers_generate-sd3.5-lora.py \
#   "${PROMPT_DIR}/evaluation_metadata.jsonl" \
#   --model "stable-diffusion-3.5-medium" \
#   --n_samples 4 --steps 32 --H 512 --W 512 --scale 6.0 \
#   --batch_size 1 --outdir "${IMAGE_FOLDER}" \
#   --lora_path sd35_lora_dpo_out/checkpoint-200 \
#   --cpu-offload


# —— 2. 评估 —— #
mkdir -p "${RESULTS_FOLDER}"
python geneval/evaluation/evaluate_images.py \
  "${IMAGE_FOLDER}" \
  --outfile "${RESULTS_FOLDER}/results_sd3.5_lora.jsonl" \
  --model-path "${OBJECT_DETECTOR_FOLDER}"

# —— 3. 汇总分数 —— #
python geneval/evaluation/summary_scores.py \
  "${RESULTS_FOLDER}/results_sd3.5_lora.jsonl"