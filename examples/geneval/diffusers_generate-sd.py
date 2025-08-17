#!/usr/bin/env python
# coding=utf-8
"""Single-GPU version for SD3.5 Medium (select GPU by --gpu, default cuda:2)"""

import argparse
import json
import os

import torch
import numpy as np
from PIL import Image
from tqdm import trange
from einops import rearrange
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor
from pytorch_lightning import seed_everything
from diffusers import (
    AutoPipelineForText2Image,
    StableDiffusion3Pipeline,  # SD3/3.5 官方管线
)

torch.set_grad_enabled(False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata_file", type=str, help="JSONL file containing lines of metadata for each prompt")
    # 指向 SD3.5 Medium 的 diffusers 目录（含 model_index.json）
    parser.add_argument("--model", type=str, default="sd3.5_medium", help="path or HF repo of SD3.5 Medium (diffusers format)")
    parser.add_argument("--outdir", type=str, default="outputs", help="dir to write results to")
    parser.add_argument("--n_samples", type=int, default=4, help="number of samples")
    parser.add_argument("--steps", type=int, default=40, help="num inference steps (SD3.5常用 28~50)")
    parser.add_argument(
        "--negative-prompt",
        type=str,
        nargs="?",
        const=("ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, "
               "out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, "
               "watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, "
               "beginner, amateur, distorted face"),
        default=None,
        help="negative prompt for guidance"
    )
    parser.add_argument("--H", type=int, default=512, help="image height, in pixel space")
    parser.add_argument("--W", type=int, default=512, help="image width, in pixel space")
    parser.add_argument("--scale", type=float, default=6.0, help="CFG scale (SD3/3.5常用 5~7)")
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument("--batch_size", type=int, default=1, help="samples produced per step")
    parser.add_argument("--skip_grid", action="store_true", help="skip saving grid")
    parser.add_argument("--gpu", type=int, default=2, help="use cuda:<gpu>, default 2")
    return parser.parse_args()


def build_pipeline(model_name_or_path: str, device: torch.device):
    """优先用 StableDiffusion3Pipeline；失败则回退 AutoPipeline。"""
    try:
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            use_safetensors=True
        )
    except Exception:
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            use_safetensors=True
        )

    # 轻量显存优化
    for fn in ("enable_vae_slicing", "enable_attention_slicing"):
        if hasattr(pipe, fn):
            try:
                getattr(pipe, fn)()
            except Exception:
                pass

    pipe = pipe.to(device)
    return pipe


def main(opt):
    # 指定 GPU
    if torch.cuda.is_available():
        assert 0 <= opt.gpu < torch.cuda.device_count(), f"--gpu 超出范围(共有 {torch.cuda.device_count()} 张卡)"
        torch.cuda.set_device(opt.gpu)
        device = torch.device(f"cuda:{opt.gpu}")
    else:
        device = torch.device("cpu")

    # 读取 prompts
    with open(opt.metadata_file) as fp:
        metadatas = [json.loads(line) for line in fp]

    # 单卡管线
    model = build_pipeline(opt.model, device)

    for index, metadata in enumerate(metadatas):
        seed_everything(opt.seed)

        outpath = os.path.join(opt.outdir, f"{index:0>5}")
        os.makedirs(outpath, exist_ok=True)

        prompt = metadata["prompt"]
        n_rows = batch_size = opt.batch_size
        print(f"Prompt ({index: >3}/{len(metadatas)}): '{prompt}'")

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
            json.dump(metadata, fp)

        sample_count = 0
        all_samples = []

        infer_kwargs = dict(
            prompt=prompt,
            height=opt.H,
            width=opt.W,
            num_inference_steps=opt.steps,
            guidance_scale=opt.scale,
            negative_prompt=(opt.negative_prompt or None),
        )

        with torch.no_grad():
            # 单卡采样
            for _ in trange((opt.n_samples + batch_size - 1) // batch_size, desc="Sampling"):
                num_imgs = min(batch_size, opt.n_samples - sample_count)
                images = model(
                    **infer_kwargs,
                    num_images_per_prompt=num_imgs,
                ).images
                for im in images:
                    im.save(os.path.join(sample_path, f"{sample_count:05}.png"))
                    sample_count += 1
                if not opt.skip_grid:
                    all_samples.append(torch.stack([ToTensor()(im) for im in images], 0))

            # 保存 grid
            if not opt.skip_grid and len(all_samples) > 0:
                grid = torch.stack(all_samples, 0)
                grid = rearrange(grid, "n b c h w -> (n b) c h w")
                grid = make_grid(grid, nrow=n_rows)
                grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()
                grid = Image.fromarray(grid.astype(np.uint8))
                grid.save(os.path.join(outpath, "grid.png"))
                del grid

        del all_samples

    print("Done.")


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
