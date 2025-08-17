#!/usr/bin/env bash
set -euo pipefail

# ============== 基本环境 ==============
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

# ============== 路径参数（按需改） ==============
# 指向“diffusers 版”的 SD3.5-medium 目录（包含 model_index.json、transformer/、vae/、text_encoder_*/ 等）
MODEL_DIR="stable-diffusion-3.5-medium"
# UnifiedReward 生成的偏好对数据。注意：其中 jpg_0 / jpg_1 必须是绝对路径
UR_JSON="sd3.5_large/baseline/RESULTS_FOLDER/data.json"
# 输出目录
OUT_DIR="sd35_lora_dpo_out"

# ============== 训练超参（3090 x2 建议） ==============
# 有效批量 = N_GPU * train_batch_size * gradient_accumulation_steps
TRAIN_BS=2
GAS=32                 # 官方示例用大累积，我们沿用
MAX_STEPS=400          # 对齐官方示例
LR=1e-7                # 对齐官方示例（有 scale_lr 时更保守）
WARMUP=40
BETA_DPO=3000
RES=512

# ============== 启动（accelerate 多卡） ==============
accelerate launch --multi_gpu --num_processes=2 UnifiedReward/DiffusionDPO/train_1.py \
  --pretrained_model_name_or_path "$MODEL_DIR" \
  --ur_data_json "$UR_JSON" \
  --output_dir "$OUT_DIR" \
  --train_batch_size $TRAIN_BS \
  --gradient_accumulation_steps $GAS \
  --max_train_steps $MAX_STEPS \
  --learning_rate $LR \
  --lr_scheduler constant_with_warmup --lr_warmup_steps $WARMUP \
  --beta_dpo $BETA_DPO \
  --lora_r 6 \
  --resolution $RES \
  --proportion_empty_prompts 0 \
  --checkpointing_steps 100 \
  --gradient_checkpointing \
  --text_encoders_on_gpu \
  --mixed_precision fp16 \
  --allow_tf32
