import torch
import os
from datasets import load_dataset, features
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, TrainingArguments
from trl import DPOConfig, DPOTrainer
from PIL import Image
import re
from dataclasses import dataclass, field
from typing import Optional
from qwen_vl_utils import process_vision_info

@dataclass
class ScriptArguments:
    model_name_or_path: str = field(metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"})
    data_path: str = field(metadata={"help": "Path to the training data."})
    image_folder: Optional[str] = field(default=None, metadata={"help": "Root folder for images"})

def main():
    from transformers import HfArgumentParser
    parser = HfArgumentParser(ScriptArguments)
    script_args, = parser.parse_args_into_dataclasses()

    # Load the model and processor for Qwen2.5-VL
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        script_args.model_name_or_path,
        torch_dtype=torch.bfloat16
    )
    ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        script_args.model_name_or_path,
        torch_dtype=torch.bfloat16
    )
    processor = AutoProcessor.from_pretrained(script_args.model_name_or_path)


    # Load the dataset
    dataset = load_dataset("json", data_files=script_args.data_path)['train']

    def format_example(example):
        # Extract image paths and load images
        image_paths = [os.path.join(script_args.image_folder, img_name) for img_name in example["images"]]
        images = [Image.open(img_path).convert("RGB") for img_path in image_paths]
        
        # 移除problem中的<image>占位符，因为Qwen2.5VL不需要它们
        clean_problem = re.sub(r'<image>', '', example["problem"]).strip()
        
        # 使用Qwen2.5VL标准消息格式
        content = []
        # 先添加图像
        for image in images:
            content.append({"type": "image", "image": image})
        # 再添加清理后的文本
        content.append({"type": "text", "text": clean_problem})
        
        # 构建消息
        prompt_messages = [{"role": "user", "content": content}]
        chosen_messages = prompt_messages + [{"role": "assistant", "content": example["chosen"]}]
        rejected_messages = prompt_messages + [{"role": "assistant", "content": example["rejected"]}]
        
        # 应用聊天模板
        prompt = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        chosen = processor.apply_chat_template(chosen_messages, tokenize=False)
        rejected = processor.apply_chat_template(rejected_messages, tokenize=False)
        
        # 处理视觉信息
        image_inputs, video_inputs = process_vision_info(prompt_messages)
        
        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "images": image_inputs if image_inputs else [],
        }

    # Apply the formatting function to the dataset
    formatted_dataset = dataset.map(format_example, remove_columns=dataset.column_names, num_proc=32)
 
    # Make sure that the images are decoded, it prevents from storing bytes.
    # More info here https://github.com/huggingface/blog/pull/2148#discussion_r1667400478
    f = formatted_dataset.features
    f["images"] = features.Sequence(features.Image(decode=True))
    formatted_dataset = formatted_dataset.cast(f)
 
    # Train the model
    training_args = DPOConfig(
        output_dir="DPO_qwenvl",
        bf16=True,
        overwrite_output_dir=True,
        gradient_checkpointing=True,
        logging_dir = './log_DPO/',
        learning_rate = 1e-6,
        use_liger_kernel=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        num_train_epochs=3,
        dataset_num_proc=32, # tokenization will use 32 processes
        dataloader_num_workers=32, # data loading will use 32 workers
        logging_steps=1,
        save_steps=100
    )

    # 训练模型
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,  # 使用PEFT时不需要参考模型
        args=training_args,
        train_dataset=formatted_dataset,
        processing_class=processor
    )

    trainer.train()

if __name__ == "__main__":
    main()