#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=2

PROMPT_DIR="geneval/prompts"
IMAGE_FOLDER="sd3.5_large/baseline/IMAGE_FOLDER_sd3.5"
RESULTS_FOLDER="sd3.5_large/baseline/RESULTS_FOLDER"
OBJECT_DETECTOR_FOLDER="geneval/<OBJECT_DETECTOR_FOLDER>"

# —— 1. 图像生成 —— #
# python geneval/generation/diffusers_generate-sd.py \
#   "${PROMPT_DIR}/evaluation_metadata.jsonl" \
#   --model "stable-diffusion-3.5-medium" \
#   --outdir "${IMAGE_FOLDER}" \

# —— 2. 评估 —— #
mkdir -p "${RESULTS_FOLDER}"
python geneval/evaluation/evaluate_images.py \
  "${IMAGE_FOLDER}" \
  --outfile "${RESULTS_FOLDER}/results_sd3.5.jsonl" \
  --model-path "${OBJECT_DETECTOR_FOLDER}"

# —— 3. 汇总分数 —— #
python geneval/evaluation/summary_scores.py \
  "${RESULTS_FOLDER}/results_sd3.5.jsonl"
