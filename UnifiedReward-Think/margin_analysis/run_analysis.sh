#!/usr/bin/env bash

export LOG_DIR="logs/margin_0"
export LOG_SAVE_FREQ=25

cd "$(dirname "$0")"

python3 qwen_margin_stats_genAI_bench_image.py
