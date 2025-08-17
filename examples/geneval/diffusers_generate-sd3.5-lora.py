#!/usr/bin/env python
# coding: utf-8
"""
SD3.5 Medium 文生图推理（多卡并行，accelerate）
- 两种并行：
  1) prompts：按 prompt 维度切分（默认）
  2) samples：同一 prompt 的 n_samples 在多卡均分（少量 prompt 更快）
- 自动加载 LoRA（目录需包含 adapter_config.json + model*.safetensors）
依赖：diffusers>=0.29, peft>=0.11, accelerate>=0.28, safetensors, einops, pytorch_lightning
"""

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
from typing import List, Tuple

import torch
from PIL import Image
from tqdm import trange
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor
from einops import rearrange
from pytorch_lightning import seed_everything
from accelerate import PartialState

from diffusers import StableDiffusion3Pipeline, AutoPipelineForText2Image
from peft import PeftModel

torch.set_grad_enabled(False)


def parse_args():
    p = argparse.ArgumentParser("SD3.5 Medium multi-GPU infer with LoRA (auto) & dual split modes")
    p.add_argument("metadata_file", type=str, help="JSONL，每行至少包含 {'prompt': ...}")
    p.add_argument("--model", type=str, required=True, help="SD3.5 Medium 的 diffusers 路径或HF仓库")
    p.add_argument("--outdir", type=str, default="outputs", help="输出目录")
    p.add_argument("--n_samples", type=int, default=4, help="每个 prompt 的总样本数")
    p.add_argument("--steps", type=int, default=40, help="推理步数（常用 28~50）")
    p.add_argument("--negative-prompt", type=str, nargs="?", const=(
        "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, "
        "out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, "
        "watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, "
        "beginner, amateur, distorted face"
    ), default=None, help="negative prompt")
    p.add_argument("--H", type=int, default=512, help="高度")
    p.add_argument("--W", type=int, default=512, help="宽度")
    p.add_argument("--scale", type=float, default=6.0, help="CFG scale（常用 5~7）")
    p.add_argument("--seed", type=int, default=42, help="基础随机种子（会叠加 rank 偏移）")
    p.add_argument("--batch_size", type=int, default=1, help="每步生成的图像数量（设备内批大小）")
    p.add_argument("--skip_grid", action="store_true", help="不保存九宫格")
    p.add_argument("--split_mode", type=str, default="prompts", choices=["prompts", "samples"],
                   help="prompts: 按 prompt 切分；samples: 同一 prompt 的 n_samples 在多卡均分")
    p.add_argument("--lora_path", type=str, default=None,
                   help="PEFT LoRA 目录（包含 adapter_config.json + model.safetensors/adapter_model.safetensors）")
    p.add_argument("--cpu-offload", action="store_true",
                   help="使用 Diffusers 的 enable_model_cpu_offload 进行推理期 CPU 卸载")
    return p.parse_args()


def load_lora_into_sd3_pipe(pipe, lora_dir, device, dtype):
    if not lora_dir:
        return pipe
    lora_dir = os.path.abspath(lora_dir)

    has_cfg = os.path.exists(os.path.join(lora_dir, "adapter_config.json"))
    has_w1  = os.path.exists(os.path.join(lora_dir, "model.safetensors"))
    has_w2  = os.path.exists(os.path.join(lora_dir, "adapter_model.safetensors"))
    if not (has_cfg and (has_w1 or has_w2)):
        raise FileNotFoundError(f"LoRA 目录缺少 adapter_config.json 或 safetensors：{lora_dir}")

    base_tf = pipe.transformer.to("cpu")
    peft_tf = PeftModel.from_pretrained(
        base_tf, lora_dir, is_trainable=False, local_files_only=True
    )
    peft_tf.to(device=device, dtype=dtype).eval()
    pipe.transformer = peft_tf
    return pipe


def build_pipeline(model_name_or_path: str, device, lora_dir: str, opt):
    # 1) 构建管线
    try:
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_name_or_path, torch_dtype=torch.float16, use_safetensors=True
        )
    except Exception:
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_name_or_path, torch_dtype=torch.float16, use_safetensors=True
        )
    # 2) 轻量省显存
    try:
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing("max")
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
    except Exception:
        pass

    # 3) LoRA + Offload / 常规放置
    if opt.cpu_offload:
        if lora_dir:
            pipe = load_lora_into_sd3_pipe(pipe, lora_dir, device="cpu", dtype=torch.float16)
            if PartialState().is_main_process:
                print(f"[LoRA] loaded (auto) from: {lora_dir} (on CPU)")
        try:
            pipe.enable_model_cpu_offload(device=device)      # 新版 diffusers
        except TypeError:
            try:
                pipe.enable_model_cpu_offload(gpu_id=device.index)  # 旧版签名
            except TypeError:
                pipe.enable_model_cpu_offload()  # 兜底：依赖当前 cuda 设备
    else:
        pipe = pipe.to(device)
        if lora_dir:
            pipe = load_lora_into_sd3_pipe(pipe, lora_dir, device=device, dtype=torch.float16)
            if PartialState().is_main_process:
                print(f"[LoRA] loaded (auto) from: {lora_dir}")
    return pipe


def slice_prompts_for_rank(metas: List[dict], rank: int, world: int):
    """返回 (global_index, meta) 列表，避免重复查找 index。"""
    return [(i, m) for i, m in enumerate(metas) if (i % world) == rank]


def even_split(n: int, world: int, rank: int):
    base = n // world
    rem = n % world
    local_n = base + (1 if rank < rem else 0)
    start = rank * base + min(rank, rem)
    return local_n, start


def save_grid_from_tensors(tensors: list, nrow: int, out_png: str, lock_path: str | None = None):
    """
    tensors: [N, 3, H, W] 的 list（每个元素是 1,3,H,W 的Tensor）
    所有 rank 可同时调用；若提供 lock_path，则用原子锁文件避免并发写。
    """
    if len(tensors) == 0:
        return
    if lock_path is not None:
        # 尝试拿锁（原子创建），失败则直接返回（说明其他 rank 在写/已写好）
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            got_lock = True
        except FileExistsError:
            got_lock = False
        if not got_lock:
            return
    try:
        grid = torch.cat(tensors, 0)            # [N,3,H,W], [0,1]
        grid = make_grid(grid, nrow=nrow)       # [3,H,W]
        grid = rearrange(grid, "c h w -> h w c")
        grid = (grid * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()
        Image.fromarray(grid, mode="RGB").save(out_png)
    finally:
        if lock_path is not None and os.path.exists(lock_path):
            try:
                os.unlink(lock_path)
            except Exception:
                pass


def run_prompts_mode(opt, pipe, state, metadatas):
    """按 prompt 切分（默认模式）"""
    os.makedirs(opt.outdir, exist_ok=True)
    local_items = slice_prompts_for_rank(metadatas, state.process_index, state.num_processes)
    base_seed = opt.seed + state.process_index

    for global_index, metadata in local_items:
        seed_everything(base_seed)
        outpath = os.path.join(opt.outdir, f"{global_index:05d}")
        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)

        prompt = metadata["prompt"]
        n_rows = batch_size = opt.batch_size
        print(f"[rank {state.process_index}] Prompt ({global_index+1}/{len(metadatas)}): '{prompt}'")

        with open(os.path.join(outpath, "metadata.jsonl"), "w", encoding="utf-8") as fp:
            fp.write(json.dumps(metadata, ensure_ascii=False))

        sample_count = 0
        all_samples = []

        infer_kwargs = dict(
            prompt=prompt,
            height=opt.H, width=opt.W,
            num_inference_steps=opt.steps,
            guidance_scale=opt.scale,
            negative_prompt=(opt.negative_prompt or None),
        )

        with torch.no_grad():
            for _ in trange((opt.n_samples + batch_size - 1) // batch_size,
                            desc=f"Sampling@rank{state.process_index}",
                            disable=not state.is_local_main_process):
                num_imgs = min(batch_size, opt.n_samples - sample_count)
                out = pipe(**infer_kwargs, num_images_per_prompt=num_imgs)
                for im in out.images:
                    im.save(os.path.join(sample_path, f"{sample_count:05d}.png"))
                    if not opt.skip_grid:
                        all_samples.append(ToTensor()(im).unsqueeze(0))
                    sample_count += 1

        if (not opt.skip_grid) and len(all_samples) > 0:
            save_grid_from_tensors(all_samples, n_rows, os.path.join(outpath, "grid.png"))

    state.wait_for_everyone()
    if state.is_main_process:
        print("Done (prompts mode).")


def run_samples_mode(opt, pipe, state, metadatas):
    """
    同一条 prompt 的 n_samples 在多卡均分。
    每个 rank 负责 local_n 张，并用 start_idx 作为文件名偏移，避免覆盖。
    所有 rank 在 barrier 后都尝试生成 grid.png，通过原子锁避免并发写。
    """
    os.makedirs(opt.outdir, exist_ok=True)
    print(f"[world_size={state.num_processes}] samples split per prompt: total_samples={opt.n_samples}")

    for global_index, metadata in enumerate(metadatas):
        seed_everything(opt.seed + state.process_index + 100003 * global_index)

        outpath = os.path.join(opt.outdir, f"{global_index:05d}")
        sample_path = os.path.join(outpath, "samples")
        if state.is_main_process:
            os.makedirs(sample_path, exist_ok=True)
        state.wait_for_everyone()

        # metadata.jsonl：原子写（已存在就跳过）
        meta_path = os.path.join(outpath, "metadata.jsonl")
        try:
            with open(meta_path, "x", encoding="utf-8") as fp:
                fp.write(json.dumps(metadata, ensure_ascii=False))
        except FileExistsError:
            pass

        prompt = metadata["prompt"]
        n_rows = opt.batch_size

        local_n, start_idx = even_split(opt.n_samples, state.num_processes, state.process_index)
        print(f"[rank {state.process_index}] Prompt ({global_index+1}/{len(metadatas)}): '{prompt}' "
              f"-> local_n={local_n}, start_idx={start_idx}")

        infer_kwargs = dict(
            prompt=prompt,
            height=opt.H, width=opt.W,
            num_inference_steps=opt.steps,
            guidance_scale=opt.scale,
            negative_prompt=(opt.negative_prompt or None),
        )

        with torch.no_grad():
            remain = local_n
            while remain > 0:
                cur_bs = min(opt.batch_size, remain)
                out = pipe(**infer_kwargs, num_images_per_prompt=cur_bs)
                for i, im in enumerate(out.images):
                    g_idx = start_idx + (local_n - remain) + i
                    fpath = os.path.join(sample_path, f"{g_idx:05d}.png")
                    im.save(fpath)
                remain -= cur_bs

        # 等全部图片写好
        state.wait_for_everyone()

        # 所有 rank 都尝试生成 grid.png（原子锁避免并发）
        if not opt.skip_grid:
            imgs = []
            for k in range(opt.n_samples):
                f = os.path.join(sample_path, f"{k:05d}.png")
                if os.path.exists(f):
                    imgs.append(ToTensor()(Image.open(f).convert("RGB")).unsqueeze(0))
            if len(imgs) > 0:
                lock_path = os.path.join(outpath, "grid.lock")
                save_grid_from_tensors(imgs, n_rows, os.path.join(outpath, "grid.png"), lock_path=lock_path)

    state.wait_for_everyone()
    if state.is_main_process:
        print("Done (samples mode).")


def main():
    opt = parse_args()
    state = PartialState()

    # 绑定本进程到对应 GPU，保证 offload 借用到本 rank 的卡
    local_rank = state.local_process_index  # 0..world-1
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    print(f"[rank{state.process_index}] bind device -> {device}, "
          f"current={torch.cuda.current_device()}, "
          f"visible={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    # 读取 prompts
    with open(opt.metadata_file, "r", encoding="utf-8") as f:
        metadatas = [json.loads(line) for line in f if line.strip()]

    # 构建管线
    pipe = build_pipeline(opt.model, device, opt.lora_path, opt)

    # 运行模式
    if opt.split_mode == "prompts":
        run_prompts_mode(opt, pipe, state, metadatas)
    else:
        run_samples_mode(opt, pipe, state, metadatas)


if __name__ == "__main__":
    main()
