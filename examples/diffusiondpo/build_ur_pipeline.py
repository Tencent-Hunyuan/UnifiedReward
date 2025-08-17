#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键 UR 闭环脚本：从已生成的图片，走 UnifiedReward 的【pair 排序 + 点评分】两路推理，
再按论文流程把结果融合为 DiffusionDPO 的 data.json。

目录结构（示例）：
  {IMAGE_ROOT}/{index}/
      ├─ metadata.jsonl   # 单条 JSON（或首行 JSONL），至少含 "prompt" 或 "caption"
      └─ samples/*.png    # 自动过滤 grid.png（stem == "grid"）

步骤：
1) 扫描 IMAGE_ROOT，构建 UR 输入：
   - point_score_input.jsonl（{"prompt": str, "image": str}）
   - pair_rank_input.jsonl  （{"prompt": str, "image0": str, "image1": str}）
   - prompts.json           （[{"images": [relpath], "problem": str}, ...]，供 vllm_inference.py 使用）
2) 调用 UR 推理脚本，得到：
   - point_scores.jsonl（image 与 score / reward）
   - pair_rank.jsonl   （image0, image1 与获胜者）
3) 融合为 DPO data.json：
   {"id": str, "caption": str, "jpg_0": str, "jpg_1": str, "label_0": 1}

新增：--action 分步执行
  - build_inputs：只构建输入与 prompts.json
  - run_ur      ：只根据现有输入运行 UR 推理
  - merge_only  ：只根据现有 UR 结果合并 data.json
  - all         ：完整流程（默认）
"""

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from itertools import combinations, islice
from collections import defaultdict

# ===== 可配置区域 =====
POINT_ITEM_KEYS = ("prompt", "image")            # 给 point_score 脚本的键
PAIR_ITEM_KEYS  = ("prompt", "image0", "image1") # 给 pair_rank  脚本的键
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
EXCLUDE_STEMS = {"grid"}                         # 过滤拼接图（文件名不含后缀）

# 仅移除这个前缀，其余路径保持不变（与 UR 推理脚本现状对齐）
PREFIX_TO_REMOVE = "/root/autodl-tmp/"

# ===== 基础工具 =====
def _abs(p: Path) -> str:
    return str(p.resolve())

def _strip_prefix(s: str) -> str:
    return s[len(PREFIX_TO_REMOVE):] if s.startswith(PREFIX_TO_REMOVE) else s

def _load_prompt_from_metadata(meta_path: Path) -> str:
    with open(meta_path, "r", encoding="utf-8") as f:
        try:
            obj = json.load(f)    # 优先单 JSON
        except Exception:
            f.seek(0)             # 兼容首行 JSONL
            line = f.readline().strip()
            obj = json.loads(line) if line else {}
    prompt = obj.get("prompt", obj.get("caption", ""))
    if not prompt:
        raise ValueError(f"{meta_path} 缺少 'prompt' 字段")
    return prompt

def _list_images(samples_dir: Path):
    return sorted([
        p for p in samples_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS and p.stem not in EXCLUDE_STEMS
    ])

def scan_outputs(root: Path):
    """
    扫描 {IMAGE_ROOT}，返回：
      records: [{"prompt":..., "image": path_after_strip, "group_id": "<index>"}]
      grouped: {prompt: [path_after_strip, ...]}
    其中 path_after_strip 为把绝对路径去掉 '/root/autodl-tmp/' 前缀后的结果，
    其他部分保持不变（例如以 'sd3.5_large/...' 开头）。
    """
    records = []
    grouped = defaultdict(list)
    for idx_dir in sorted(root.iterdir()):
        if not idx_dir.is_dir():
            continue
        meta = idx_dir / "metadata.jsonl"
        samples = idx_dir / "samples"
        if not meta.exists() or not samples.exists():
            continue
        prompt = _load_prompt_from_metadata(meta)
        imgs = _list_images(samples)
        for img in imgs:
            full_path = _abs(img)
            stripped = _strip_prefix(full_path)   # —— 仅去掉 /root/autodl-tmp/ 前缀
            rec = {"prompt": prompt, "image": stripped, "group_id": idx_dir.name}
            records.append(rec)
            grouped[prompt].append(stripped)
    if not records:
        raise RuntimeError(f"在 {root} 未发现有效的 (metadata.jsonl + samples/*) 结构")
    return records, grouped

def write_jsonl(items, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"[OK] 写 JSONL：{out_path}（{len(items)} 行）")

def write_json(items, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"[OK] 写 JSON：{out_path}（{len(items)} 条）")

def _bounded_pairs(lst, max_pairs=None):
    gen = combinations(lst, 2)
    return gen if not max_pairs or max_pairs <= 0 else islice(gen, max_pairs)

# ===== 构建 UR 输入 =====
def make_point_input(image_root: Path, out_path: Path):
    records, _ = scan_outputs(image_root)
    k_prompt, k_image = POINT_ITEM_KEYS
    items = [{k_prompt: r["prompt"], k_image: r["image"]} for r in records]
    write_jsonl(items, out_path)
    return out_path

def make_pair_input(image_root: Path, out_path: Path, max_pairs_per_prompt: int = None):
    _, grouped = scan_outputs(image_root)
    k_prompt, k0, k1 = PAIR_ITEM_KEYS
    items = []
    for prompt, paths in grouped.items():
        for a, b in _bounded_pairs(paths, max_pairs_per_prompt):
            items.append({k_prompt: prompt, k0: a, k1: b})
    write_jsonl(items, out_path)
    return out_path

def make_prompts_json(image_root: Path, out_path: Path):
    """
    生成给 vllm_inference.py 的 prompts.json
    与 point/pair 的路径风格完全一致：
    "images": ["sd3.5_large/baseline/IMAGE_FOLDER/00003/samples/00002.png"]
    """
    records, _ = scan_outputs(image_root)
    items = []
    for r in records:
        path_str = r["image"]  # 已是去前缀后的最终形式
        items.append({
            "images": [path_str],
            "problem": r["prompt"]
        })
    write_json(items, out_path)
    return out_path



# ===== 运行 UR 推理 =====
def run_cmd(cmd: str | list):
    if isinstance(cmd, str):
        cmd_list = shlex.split(cmd)
    else:
        cmd_list = [str(x) for x in cmd]
    print("[RUN]", " ".join(cmd_list))
    res = subprocess.run(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr)
        raise RuntimeError(f"命令执行失败：{' '.join(cmd_list)}")
    return res

# ===== 读取 UR 结果 =====
def load_point_scores(scores_jsonl: Path):
    """
    读取点评分结果（JSONL），兼容：
      {"image": <path>, "score": <float>}
      {"image_path": <path>, "reward": <float>}
    注意：这里假定 UR 脚本回写的 <path> 与输入一致（即已去掉 '/root/autodl-tmp/' 前缀）。
    """
    mp = {}
    with open(scores_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            img = o.get("image") or o.get("image_path")
            sc  = o.get("score")
            if sc is None:
                sc = o.get("reward")
            if img is None or sc is None:
                continue
            mp[img] = float(sc)
    if not mp:
        raise RuntimeError(f"{scores_jsonl} 中未解析到 (image, score)")
    return mp

def load_pair_rank(pair_jsonl: Path):
    """
    读取成对排序结果，统计每张图片的胜场（wins）。
    兼容键：
      - winner 路径：chosen / winner_image
      - winner 索引：chosen_idx / preferred_index / winner / label ∈ {0,1}
    """
    wins = defaultdict(int)
    with open(pair_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            img0 = o.get("image0"); img1 = o.get("image1")
            if not img0 or not img1:
                continue
            chosen_path = o.get("chosen") or o.get("winner_image")
            chosen_idx  = o.get("chosen_idx")
            if chosen_idx is None:
                chosen_idx = o.get("preferred_index", o.get("winner", o.get("label")))
            winner = None
            if chosen_path in (img0, img1):
                winner = chosen_path
            elif chosen_idx in (0,1,"0","1"):
                winner = img0 if int(chosen_idx) == 0 else img1
            if winner == img0:
                wins[img0] += 1
            elif winner == img1:
                wins[img1] += 1
    return wins

# ===== 融合为 DPO data.json =====
def build_dpo_point_only(image_root: Path, point_scores_jsonl: Path, out_json: Path, topk=1, bottomk=1):
    _, grouped = scan_outputs(image_root)
    scores = load_point_scores(point_scores_jsonl)

    items = []
    pid = 0
    for prompt, paths in grouped.items():
        scored = [(p, scores[p]) for p in paths if p in scores]
        if len(scored) < 2:
            continue
        scored.sort(key=lambda x: x[1], reverse=True)
        tops = [p for p,_ in scored[:max(1, topk)]]
        bots = [p for p,_ in scored[-max(1, bottomk):]]
        for a in tops:
            for b in bots:
                if a == b:
                    continue
                items.append({"id": str(pid), "caption": prompt, "jpg_0": a, "jpg_1": b, "label_0": 1})
                pid += 1

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"[OK] DPO data（point-only）：{out_json}（{len(items)} 对）")

def build_dpo_pair_then_point(image_root: Path, pair_rank_jsonl: Path, point_scores_jsonl: Path,
                              out_json: Path, top_m=2, bottom_m=2, final_topk=1, final_bottomk=1):
    """
    先用 pair 排序选候选（按 wins 取 top_m / bottom_m），
    再在候选里用点评分取极值（final_topk / final_bottomk），形成 (优,劣) 对。
    """
    _, grouped = scan_outputs(image_root)
    wins = load_pair_rank(pair_rank_jsonl)
    scores = load_point_scores(point_scores_jsonl)

    items = []
    pid = 0
    for prompt, paths in grouped.items():
        if len(paths) < 2:
            continue
        lst = [(p, wins.get(p, 0)) for p in paths]
        lst.sort(key=lambda x: x[1], reverse=True)
        cand_top = [p for p,_ in lst[:max(1, top_m)]]
        cand_bot = [p for p,_ in lst[-max(1, bottom_m):]]

        cand_top_scored = sorted([(p, scores.get(p, float("-inf"))) for p in cand_top], key=lambda x: x[1], reverse=True)
        cand_bot_scored = sorted([(p, scores.get(p, float("inf")))  for p in cand_bot], key=lambda x: x[1])

        final_tops = [p for p,_ in cand_top_scored[:max(1, final_topk)]]
        final_bots = [p for p,_ in cand_bot_scored[:max(1, final_bottomk)]]

        for a in final_tops:
            for b in final_bots:
                if a == b:
                    continue
                items.append({"id": str(pid), "caption": prompt, "jpg_0": a, "jpg_1": b, "label_0": 1})
                pid += 1

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"[OK] DPO data（pair→point）：{out_json}（{len(items)} 对）")

# ===== CLI =====
def main():
    ap = argparse.ArgumentParser(description="UnifiedReward 全流程：构建输入 -> 运行 UR 推理 -> 生成 DPO data.json（过滤 grid.png）")

    # 基本路径
    ap.add_argument("--image_root", type=Path, required=True, help="你的生成图片根目录，如 sd3.5_large/baseline/IMAGE_FOLDER")
    ap.add_argument("--results_dir", type=Path, required=True, help="UR 输入与输出保存目录，如 sd3.5_large/baseline/RESULTS_FOLDER")

    # UR 仓库与脚本路径
    ap.add_argument("--ur_repo", type=Path, required=True, help="UnifiedReward 仓库根目录")
    ap.add_argument("--point_script", type=str, default="inference_qwen/image_generation/qwen_point_score_image_generation.py")
    ap.add_argument("--pair_script",  type=str, default="inference_qwen/image_generation/qwen_pair_rank_image_generation.py")

    # 额外参数（原样拼接给 UR 脚本，例如：--model UnifiedReward-qwen-7b --batch_size 8）
    ap.add_argument("--point_extra_args", type=str, default="")
    ap.add_argument("--pair_extra_args",  type=str, default="")

    # pair 输入限流
    ap.add_argument("--max_pairs_per_prompt", type=int, default=50)

    # 执行动作：分步 or 全流程
    ap.add_argument("--action", choices=["build_inputs", "run_ur", "merge_only", "all"], default="all")

    # DPO 融合策略
    ap.add_argument("--strategy", choices=["point_only", "pair_then_point"], default="pair_then_point")
    ap.add_argument("--topk", type=int, default=1, help="[point_only] 每个 prompt 取最高分 topk")
    ap.add_argument("--bottomk", type=int, default=1, help="[point_only] 每个 prompt 取最低分 bottomk")
    ap.add_argument("--top_m", type=int, default=2, help="[pair_then_point] wins 前 top_m 进候选")
    ap.add_argument("--bottom_m", type=int, default=2, help="[pair_then_point] wins 后 bottom_m 进候选")
    ap.add_argument("--final_topk", type=int, default=1, help="[pair_then_point] 候选里用 point 取最高 final_topk")
    ap.add_argument("--final_bottomk", type=int, default=1, help="[pair_then_point] 候选里用 point 取最低 final_bottomk")

    args = ap.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    # 路径
    point_in = args.results_dir / "point_score_input.jsonl"
    pair_in  = args.results_dir / "pair_rank_input.jsonl"
    prompts  = args.results_dir / "prompts.json"
    point_out = args.results_dir / "point_scores.jsonl"
    pair_out  = args.results_dir / "pair_rank.jsonl"
    dpo_out   = args.results_dir / "data.json"

    # === 1) 构建输入（含 prompts.json） ===
    def _do_build_inputs():
        make_point_input(args.image_root, point_in)
        make_pair_input(args.image_root, pair_in, args.max_pairs_per_prompt)
        make_prompts_json(args.image_root, prompts)

    # === 2) 调 UR 推理 ===
    def _do_run_ur():
        point_cmd = [
            "python", str(args.ur_repo / args.point_script),
            "--input", str(point_in),
            "--output", str(point_out)
        ]
        pair_cmd = [
            "python", str(args.ur_repo / args.pair_script),
            "--input", str(pair_in),
            "--output", str(pair_out)
        ]
        if args.point_extra_args:
            point_cmd += shlex.split(args.point_extra_args)
        if args.pair_extra_args:
            pair_cmd  += shlex.split(args.pair_extra_args)
        run_cmd(point_cmd)
        run_cmd(pair_cmd)

    # === 3) 合流生成 DPO 数据 ===
    def _do_merge_only():
        if args.strategy == "point_only":
            build_dpo_point_only(args.image_root, point_out, dpo_out, topk=args.topk, bottomk=args.bottomk)
        else:
            build_dpo_pair_then_point(
                args.image_root, pair_out, point_out, dpo_out,
                top_m=args.top_m, bottom_m=args.bottom_m,
                final_topk=args.final_topk, final_bottomk=args.final_bottomk
            )

    # === 执行控制 ===
    if args.action == "build_inputs":
        _do_build_inputs()
        return
    if args.action == "run_ur":
        _do_run_ur()
        return
    if args.action == "merge_only":
        _do_merge_only()
        return

    # all
    _do_build_inputs()
    _do_run_ur()
    _do_merge_only()

if __name__ == "__main__":
    main()
