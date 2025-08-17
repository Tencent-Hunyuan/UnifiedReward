#!/usr/bin/env python
# coding: utf-8
"""
SD3.5-medium LoRA + DPO (Diffusers) — 稳定维度对齐版
- 单 Transformer + LoRA 开/关 充当 online/ref，显存小
- FlowMatch：set_timesteps + scale_noise（target = noise）
- 结束时输出每个 checkpoint 的 loss，并将最终导出的 lora_adapter
  回滚到 loss 最低的 checkpoint 再导出
"""

import os, gc, math, random, argparse, json, shutil
from typing import List, Dict, Any, Tuple, Optional
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from datasets import load_dataset
from torchvision import transforms
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from peft import LoraConfig, get_peft_model

from diffusers import (
    StableDiffusion3Pipeline,
    SD3Transformer2DModel,
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.optimization import get_scheduler

logger = get_logger(__name__, log_level="INFO")

# ------------------ args ------------------
def parse_args():
    p = argparse.ArgumentParser("SD3.5 LoRA + DPO (shape-safe)")
    p.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    p.add_argument("--ur_data_json", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="sd35_dpo_out")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mixed_precision", type=str, default="fp16", choices=["no","fp16","bf16"])
    p.add_argument("--allow_tf32", action="store_true")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--random_crop", action="store_true")
    p.add_argument("--no_hflip", action="store_true")
    p.add_argument("--proportion_empty_prompts", type=float, default=0.0)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=128)
    p.add_argument("--dataloader_num_workers", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=1e-7)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.999)
    p.add_argument("--adam_weight_decay", type=float, default=1e-2)
    p.add_argument("--adam_epsilon", type=float, default=1e-8)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--lr_scheduler", type=str, default="constant_with_warmup",
                   choices=["linear","cosine","cosine_with_restarts","polynomial","constant","constant_with_warmup"])
    p.add_argument("--lr_warmup_steps", type=int, default=40)
    p.add_argument("--max_train_steps", type=int, default=400)
    p.add_argument("--beta_dpo", type=float, default=3000.0)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--target_modules", type=str, nargs="*", default=["to_q","to_v"])
    p.add_argument("--merge_lora", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--checkpointing_steps", type=int, default=100)
    p.add_argument("--report_to", type=str, default="tensorboard")
    p.add_argument("--logging_dir", type=str, default="logs")
    p.add_argument("--tracker_project_name", type=str, default="sd35_dpo_shapesafe")
    p.add_argument("--text_encoders_on_gpu", action="store_true")
    args = p.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1:
        setattr(args, "local_rank", env_local_rank)
    return args

# ------------------ data ------------------
def build_transforms(args):
    t = [transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR)]
    t.append(transforms.RandomCrop(args.resolution) if args.random_crop else transforms.CenterCrop(args.resolution))
    if not args.no_hflip: t.append(transforms.RandomHorizontalFlip())
    t += [transforms.ToTensor(), transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])]
    return transforms.Compose(t)

def make_loader(args, tform):
    ds_all = load_dataset("json", data_files=args.ur_data_json)
    split = list(ds_all.keys())[0]
    def preprocess(ex):
        caps, pair = [], []
        for cap, p0, p1, l0 in zip(ex.get("caption",[]), ex.get("jpg_0",[]), ex.get("jpg_1",[]), ex.get("label_0",[])):
            if isinstance(cap, list): cap = random.choice(cap) if cap else ""
            if random.random() < args.proportion_empty_prompts: cap = ""
            caps.append(str(cap) if cap is not None else "")
            im0 = Image.open(p0).convert("RGB"); im1 = Image.open(p1).convert("RGB")
            t0 = tform(im0); t1 = tform(im1)
            win, lose = (t0,t1) if int(l0)==1 else (t1,t0)
            pair.append(torch.stack([win, lose], dim=0))
        return {"caption": caps, "pair_pixels": pair}
    ds = ds_all[split]
    if args.max_train_samples is not None:
        ds = ds.shuffle(seed=args.seed).select(range(args.max_train_samples))
    train_ds = ds.with_transform(preprocess)
    def collate(examples: List[Dict[str,Any]]):
        return {
            "pair_pixels": torch.stack([e["pair_pixels"] for e in examples]),
            "caption": [e["caption"] for e in examples],
        }
    return DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True,
                      num_workers=args.dataloader_num_workers, pin_memory=True, drop_last=True,
                      collate_fn=collate)

# ------------------ helpers ------------------
def adapt_last_dim(x: torch.Tensor, target: Optional[int]) -> torch.Tensor:
    if target is None: return x
    last = x.shape[-1]
    if last == target: return x
    if last == 2*target:
        a = x[..., :target]; b = x[..., target:2*target]
        return 0.5*(a+b)
    if last > target:
        return x[..., :target]
    return F.pad(x, (0, target-last))

# ------------------ main ------------------
def main():
    args = parse_args()
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=ProjectConfiguration(project_dir=args.output_dir,
                                            logging_dir=os.path.join(args.output_dir, args.logging_dir)),
    )
    if accelerator.is_main_process: os.makedirs(args.output_dir, exist_ok=True)
    if args.seed is not None: set_seed(args.seed + accelerator.process_index)

    dtype = torch.float16 if accelerator.mixed_precision=="fp16" else (torch.bfloat16 if accelerator.mixed_precision=="bf16" else torch.float32)

    # ----- Pipeline (只用于文本编码) -----
    pipe = StableDiffusion3Pipeline.from_pretrained(args.pretrained_model_name_or_path)
    if args.text_encoders_on_gpu:
        for name in ["text_encoder","text_encoder_2","text_encoder_3"]:
            te = getattr(pipe, name, None)
            if te is not None: te.to(accelerator.device, dtype=dtype)

    # ----- VAE -----
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    vae.requires_grad_(False); vae.to(accelerator.device, dtype=dtype)
    vae.enable_slicing(); vae.enable_tiling()

    # ----- Scheduler -----
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    n_ts = getattr(scheduler.config, "num_train_timesteps", None) or 1000
    try: scheduler.set_timesteps(n_ts, device=accelerator.device)
    except TypeError: scheduler.set_timesteps(n_ts)

    # ----- Transformer + LoRA -----
    transformer = SD3Transformer2DModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="transformer")
    CROSS_DIM  = getattr(transformer.config, "cross_attention_dim", 1536)
    POOLED_DIM = getattr(transformer.config, "pooled_projection_dim", None) or getattr(transformer.config, "pooled_dim", 2048)
    if accelerator.is_main_process:
        print(f"[expect] cross_attention_dim={CROSS_DIM}, pooled_dim={POOLED_DIM}")

    lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                      bias="none", target_modules=args.target_modules)
    transformer = get_peft_model(transformer, lcfg)
    if args.gradient_checkpointing and hasattr(transformer, "enable_gradient_checkpointing"):
        try: transformer.enable_gradient_checkpointing()
        except Exception: pass

    # ----- Optim / Data / Sched -----
    optim = torch.optim.AdamW(transformer.parameters(), lr=args.learning_rate,
                              betas=(args.adam_beta1,args.adam_beta2),
                              weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)
    loader = make_loader(args, build_transforms(args))
    lr_sched = get_scheduler(args.lr_scheduler, optimizer=optim,
                             num_warmup_steps=args.lr_warmup_steps*accelerator.num_processes,
                             num_training_steps=args.max_train_steps*accelerator.num_processes)

    transformer, optim, loader, lr_sched = accelerator.prepare(transformer, optim, loader, lr_sched)

    # LoRA on/off
    def set_lora(active: bool):
        base = accelerator.unwrap_model(transformer)
        if hasattr(base,"enable_adapter") and hasattr(base,"disable_adapter"):
            base.enable_adapter() if active else base.disable_adapter()
        elif hasattr(base,"enable_adapter_layers") and hasattr(base,"disable_adapter_layers"):
            base.enable_adapter_layers() if active else base.disable_adapter_layers()
        else:
            for m in base.modules():
                if hasattr(m,"enable_adapters") and active: m.enable_adapters()
                if hasattr(m,"disable_adapters") and not active: m.disable_adapters()

    # ----- encode_prompt：一次性对齐维度 -----
    @torch.no_grad()
    def encode_prompts(caps: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        dev_enc = accelerator.device if args.text_encoders_on_gpu else "cpu"
        try:
            out = pipe.encode_prompt(
                prompt=caps, prompt_2=caps, prompt_3=caps,
                device=dev_enc, num_images_per_prompt=1
            )
        except TypeError:
            out = pipe.encode_prompt(prompt=caps, device=dev_enc, num_images_per_prompt=1)

        if isinstance(out, dict):
            pe = out["prompt_embeds"]      # [B, T, 4096]
            pooled = out.get("pooled_prompt_embeds", None) or out.get("pooled_projections", None)
        else:
            pe, pooled = out[0], None

        if pooled is None:
            pooled = pe.mean(dim=1)
        elif pooled.ndim == 3:
            pooled = pooled.mean(dim=1)

        pe = pe.to(accelerator.device, dtype=dtype)
        pooled = pooled.to(accelerator.device, dtype=dtype)
        pooled = adapt_last_dim(pooled, POOLED_DIM)
        return pe, pooled

    def add_noise_and_target(latents, t, noise):
        return scheduler.scale_noise(latents, t, noise), noise

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, vars(args))

    # ====== 新增：记录 checkpoint 的 loss ======
    ckpt_losses: Dict[int, float] = {}   # step -> loss
    # =========================================

    global_step = 0
    progress = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress.set_description("Steps")

    first_log = True
    while global_step < args.max_train_steps:
        for batch in loader:
            transformer.train()
            with accelerator.accumulate(transformer):
                pv = batch["pair_pixels"].to(accelerator.device)  # (B,2,C,H,W)
                B, two, C, H, W = pv.shape
                assert two == 2
                imgs = pv.view(B*2, C, H, W)

                with torch.no_grad():
                    latents = vae.encode(imgs.to(dtype=dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                noise_pair = noise.view(B,2,*noise.shape[1:])
                noise_pair[:,1] = noise_pair[:,0]
                noise = noise_pair.view_as(noise)

                idx = torch.randint(0, scheduler.timesteps.shape[0], (B,), device=latents.device)
                t = scheduler.timesteps.index_select(0, idx)
                timesteps = t.repeat_interleave(2)

                pe_B, pooled_B = encode_prompts(batch["caption"])
                if first_log and accelerator.is_main_process:
                    print(f"[debug] after-fit prompt_embeds: {pe_B.shape}, pooled: {pooled_B.shape}")
                    first_log = False
                pe = pe_B.repeat_interleave(2, dim=0)
                pooled = pooled_B.repeat_interleave(2, dim=0)

                noisy, target = add_noise_and_target(latents, timesteps, noise)

                win = torch.arange(0, 2*B, 2, device=latents.device)
                lose = win + 1

                def fwd(use_lora: bool, sel: torch.Tensor):
                    set_lora(use_lora)
                    return transformer(
                        hidden_states=noisy.index_select(0, sel),
                        timestep=timesteps.index_select(0, sel),
                        encoder_hidden_states=pe.index_select(0, sel),
                        pooled_projections=pooled.index_select(0, sel),
                    ).sample

                with torch.no_grad():
                    ref_w = fwd(False, win); ref_l = fwd(False, lose)
                model_w = fwd(True, win);  model_l = fwd(True, lose)

                tgt_w = target.index_select(0, win)
                tgt_l = target.index_select(0, lose)
                mw = (model_w - tgt_w).pow(2).mean(dim=[1,2,3])
                ml = (model_l - tgt_l).pow(2).mean(dim=[1,2,3])
                rw = (ref_w   - tgt_w).pow(2).mean(dim=[1,2,3])
                rl = (ref_l   - tgt_l).pow(2).mean(dim=[1,2,3])
                inside = -0.5 * args.beta_dpo * ((mw-ml) - (rw-rl))
                loss = -F.logsigmoid(inside).mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                optim.step(); lr_sched.step(); optim.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                # 记录当前 step 的训练 loss
                step_loss = float(loss.detach().item())
                if accelerator.is_main_process:
                    accelerator.log({"train/dpo_loss": step_loss,
                                     "lr": lr_sched.get_last_lr()[0]}, step=global_step)
                progress.update(1); progress.set_postfix({"loss": f"{step_loss:.4f}"})

                # 到达 checkpoint 步，保存并记录该 checkpoint 的 loss
                if (global_step % args.checkpointing_steps == 0) and accelerator.is_main_process:
                    ckpt = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(ckpt, exist_ok=True)
                    accelerator.save_state(ckpt)
                    ckpt_losses[global_step] = step_loss
                    logger.info(f"[checkpoint] saved: {ckpt}  | loss={step_loss:.6f}")

                if global_step >= args.max_train_steps:
                    break
        gc.collect(); torch.cuda.empty_cache()

    accelerator.wait_for_everyone()

    # ====== 训练结束：选择最优 checkpoint 并回滚保存 LoRA ======
    best_step: Optional[int] = None
    if accelerator.is_main_process:
        # 若没有触发过 checkpoint（训练步数太少），不做选择，直接保存当前
        if len(ckpt_losses) > 0:
            # 落盘所有 checkpoint 的 loss
            with open(os.path.join(args.output_dir, "checkpoint_losses.json"), "w", encoding="utf-8") as f:
                json.dump({str(k): float(v) for k, v in sorted(ckpt_losses.items())}, f, indent=2, ensure_ascii=False)

            # 打印汇总
            logger.info("===== Checkpoint Losses =====")
            for s, v in sorted(ckpt_losses.items()):
                logger.info(f"checkpoint-{s}: loss={v:.6f}")

            # 选最低
            best_step = min(ckpt_losses, key=lambda s: ckpt_losses[s])
            best_loss = ckpt_losses[best_step]
            with open(os.path.join(args.output_dir, "best_checkpoint.txt"), "w", encoding="utf-8") as f:
                f.write(f"best_step={best_step}\nloss={best_loss:.8f}\n")
            logger.info(f"[best] step={best_step}, loss={best_loss:.6f}")
        else:
            logger.info("[best] 未产生任何 checkpoint（未到达 checkpointing_steps），将使用最终权重导出。")

    # 广播同步 best_step 给所有进程
    if accelerator.num_processes > 1:
        # 用 CPU 张量广播
        t = torch.tensor([-1 if best_step is None else best_step], dtype=torch.long, device=accelerator.device)
        accelerator.broadcast(t, src=0)
        best_step = None if int(t.item()) < 0 else int(t.item())

    # 如有最优 checkpoint，回滚到该状态
    if best_step is not None:
        best_dir = os.path.join(args.output_dir, f"checkpoint-{best_step}")
        accelerator.load_state(best_dir)
        accelerator.wait_for_everyone()

    base = accelerator.unwrap_model(transformer)
    if accelerator.is_main_process:
        lora_dir = os.path.join(args.output_dir, "lora_adapter")
        # 覆盖为“最佳 checkpoint（若存在）或最终权重”的 LoRA
        if os.path.isdir(lora_dir):
            shutil.rmtree(lora_dir)
        os.makedirs(lora_dir, exist_ok=True)
        base.save_pretrained(lora_dir)
        if best_step is not None:
            logger.info(f"[save] LoRA adapter (from best checkpoint-{best_step}) -> {lora_dir}")
            # 方便定位：创建/更新一个软链接指向最佳 checkpoint（若系统支持）
            best_alias = os.path.join(args.output_dir, "checkpoint-best")
            try:
                if os.path.islink(best_alias) or os.path.exists(best_alias):
                    try:
                        os.remove(best_alias)
                    except IsADirectoryError:
                        shutil.rmtree(best_alias)
                os.symlink(os.path.basename(best_dir), best_alias)
            except Exception:
                # 某些环境不支持软链接，忽略
                pass
        else:
            logger.info(f"[save] LoRA adapter (from final weights) -> {lora_dir}")

    accelerator.end_training()

if __name__ == "__main__":
    main()
