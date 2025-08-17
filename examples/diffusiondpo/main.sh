#!/usr/bin/env bash
set -euo pipefail

# 固定使用 GPU 2（按需改）
export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

IMAGE_FOLDER="sd3.5_large/baseline/IMAGE_FOLDER"
RESULTS_FOLDER="sd3.5_large/baseline/RESULTS_FOLDER"
UR_DIR="UnifiedReward"
UR_MODEL="UnifiedReward-qwen-7b"
BATCH=8

# 1️⃣ 只构建 UR 输入 + 生成 prompts.json（不跑推理）
python sd3.5_large/build_ur_pipeline.py \
  --image_root "${IMAGE_FOLDER}" \
  --results_dir "${RESULTS_FOLDER}" \
  --ur_repo "${UR_DIR}" \
  --action build_inputs \
  --strategy pair_then_point \
  --max_pairs_per_prompt 50

# 跑你的 vLLM 评测；prompts.json 已由上一步生成
python UnifiedReward/vllm_qwen/vllm_inference.py \
  --api_url http://127.0.0.1:8080 \
  --prompt_path "${RESULTS_FOLDER}/prompts.json" \
  --image_root "${IMAGE_FOLDER}" \
  --output_path "${RESULTS_FOLDER}/ur_results.json" \
  --max_workers 4

# 2️⃣  UR point / pair 推理
python ${UR_DIR}/inference_qwen/image_generation/qwen_point_score_image_generation.py \
  --input  "${RESULTS_FOLDER}/point_score_input.jsonl" \
  --output "${RESULTS_FOLDER}/point_scores.jsonl" \
  --model  "${UR_MODEL}" \
  --batch_size ${BATCH}

# 释放显存
python - <<'PY'
import torch, gc; torch.cuda.empty_cache(); gc.collect(); print("[OK] freed CUDA cache")
PY

python ${UR_DIR}/inference_qwen/image_generation/qwen_pair_rank_image_generation.py \
  --input  "${RESULTS_FOLDER}/pair_rank_input.jsonl" \
  --output "${RESULTS_FOLDER}/pair_rank.jsonl" \
  --model  "${UR_MODEL}" \
  --batch_size ${BATCH}

# 3️⃣ 仅合并生成 DPO 数据（不重复前两步）
python sd3.5_large/build_ur_pipeline.py \
  --image_root "${IMAGE_FOLDER}" \
  --results_dir "${RESULTS_FOLDER}" \
  --ur_repo "${UR_DIR}" \
  --action merge_only \
  --strategy pair_then_point \
  --top_m 2 --bottom_m 2 --final_topk 1 --final_bottomk 1

echo "✅ 完成 DPO 数据构建：${RESULTS_FOLDER}/data.json"
