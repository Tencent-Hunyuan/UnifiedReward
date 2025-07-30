import asyncio
import json
import os
import re
import base64
import mimetypes
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Dict, Tuple, Any, List
from openai import AsyncOpenAI
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from queue import Queue

# --- 阶段三：CPU密集型函数 (不变) ---
# 对预测结果进行判断
def process_and_format_result(data_to_process: Dict) -> Tuple[bool, Dict]:

    output_text = data_to_process['output_text']
    prompt = data_to_process['prompt_text_for_saving']
    ori_item = data_to_process['ori_item']
    truth_answer = data_to_process['truth']
    
    answer_pattern = r'<answer>(.*?)</answer>'
    model_answer_match = re.search(answer_pattern, output_text, re.DOTALL)
    model_answer = model_answer_match.group(1).strip() if model_answer_match else output_text
    
    winner_phrase = truth_answer.split(" than ", 1)[0].strip('.')
    is_correct = winner_phrase is not None and winner_phrase in model_answer
    
    if is_correct:
        result_json = {
            'conversations': [
                {'from': 'human', 'value': prompt},
                {'from': 'gpt', 'value': output_text}
            ], 'images': ori_item.get('images', [])
        }
    else:
        result_json = {
            'conversations': [
                {'from': 'human', 'value': prompt},
                {'from': 'gpt', 'value': output_text}
            ], 'images': ori_item.get('images', [])
        }
        # {
        #     'problem': prompt, 'solution': winner_phrase,
        #     'images': ori_item.get('images', [])
        # }
    return is_correct, result_json

# --- 阶段一：辅助函数 ---
def encode_image_to_base64(image_path: str) -> str:
    # (此函数逻辑不变)
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None: mime_type = 'application/octet-stream'
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded_string}"

# --- 阶段一：数据集 & 高度优化的 collate_fn ---
# 读取数据集，修改读取方式，即可读取不同的数据集，将不同阶段进行解耦
class HPDDataset(Dataset):
    # (Dataset 本身基本不变，只返回路径和文本)
    def __init__(self, json_file_path, base_path):
        self.base_path = base_path
        with open(json_file_path, 'r', encoding='utf-8') as f:
            self.data_index = json.load(f)

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        data_item = self.data_index[idx]
        image_paths = data_item.get('images', [])
        prompt_text = data_item.get('prompt', [])
        truth_answer = next((conv.get('value', '') for conv in data_item.get('conversations', []) if conv.get('from') == 'gpt'), "")
        
        if not prompt_text or len(image_paths) < 2: return None
        
        abs_paths = [os.path.join(self.base_path, p) for p in image_paths]
        if not all(os.path.exists(p) for p in abs_paths): return None

        return {"image_paths": abs_paths[:2], "prompt": prompt_text, "truth": truth_answer, "ori_item": data_item}

## 对于不同数据集，在这个部分进行修改实现不同数据的预处理以生成符合的数据
class CollateFn:
    def __init__(self, prompt_template: str):
        """在初始化时接收模板，并将其存储为实例变量。"""
        self.prompt_template = prompt_template

    def __call__(self, batch: List[Dict]) -> List[Dict]:
        """
        实现__call__方法，使类的实例可以像函数一样被调用。
        这里的逻辑与您之前的collate_fn完全相同。
        """
        processed_batch = []
        for item in batch:
            if item is None: continue
            
            try:
                b64_images = [encode_image_to_base64(p) for p in item["image_paths"]]
            except Exception:
                continue

            messages = [
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": b64_images[0]}},
                    {"type": "image_url", "image_url": {"url": b64_images[1]}},
                    # 使用 self.prompt_template 访问模板
                    {"type": "text", "text": self.prompt_template.format(prompt=item["prompt"])},
                ]}
            ]
            
            processed_batch.append({
                "messages": messages,
                "prompt_text_for_saving": f"<image><image>{self.prompt_template.format(prompt=item["prompt"])}",
                "truth": item["truth"],
                "ori_item": item["ori_item"],
            })
        return processed_batch

# --- 2. 简化生产者 ---
def producer(queue: Queue, loader: DataLoader):
    """使用标准的、线程安全的 queue.Queue。"""
    print("Producer thread started.")
    for batch in loader:
        for item in batch:
            queue.put(item)
    print("Producer thread finished.")


# --- 3. 改造消费者 (Workers) ---
async def inference_worker(
    client: AsyncOpenAI,
    request_queue: Queue,
    result_queue: Queue,
    loop: asyncio.AbstractEventLoop,
    io_executor: ThreadPoolExecutor,
    model_name: str,
    sampling_params: Dict,
    pbar: tqdm,
):
    while True:
        item = await loop.run_in_executor(io_executor, request_queue.get)

        if item is None:
            break

        try:
            stream = await client.chat.completions.create(
                model=model_name, messages=item['messages'], stream=True, **sampling_params,
            )
            full_response_content = "".join([chunk.choices[0].delta.content async for chunk in stream if chunk.choices[0].delta.content is not None])

            if full_response_content:
                result_data = item.copy()
                result_data['output_text'] = full_response_content
                del result_data['messages']
                await loop.run_in_executor(io_executor, result_queue.put, result_data)

        except Exception as e:
            print(f"\nInference error: {type(e).__name__}: {e}")
        finally:
            pbar.update(1)

# --- 阶段三：结果处理 Worker ---
async def result_worker(
    result_queue: Queue,
    loop: asyncio.AbstractEventLoop,
    io_executor: ThreadPoolExecutor,
    cpu_executor: ProcessPoolExecutor,
    f_correct,
    f_wrong,
    pbar: tqdm,
):
    while True:
        result_data = await loop.run_in_executor(io_executor, result_queue.get)
        
        if result_data is None:
            break

        try:
            is_correct, result_json = await loop.run_in_executor(
                cpu_executor, process_and_format_result, result_data
            )
            
            output_str = json.dumps(result_json, ensure_ascii=False) + '\n'
            file_to_write = f_correct if is_correct else f_wrong

            await loop.run_in_executor(io_executor, file_to_write.write, output_str)
        except Exception as e:
            print(f"\nResult processing error: {type(e).__name__}: {e}")
        finally:
            pbar.update(1)

# --- 主函数：编排整个三阶段流水线 ---
async def main():
    # --- 配置 ---，可修改
    OPENAI_API_BASE = "http://127.0.0.1:8000/v1"
    MODEL_NAME = "/root/UnifiedReward-main/UnifiedReward-Think/UnifiedReward-qwen-7b"
    base_path = '/root/UnifiedReward-main/UnifiedReward-Think/dataset/EvalMuse/'
    data_file_path = os.path.join(base_path, 'pairwise/train_data.json')
    output_dir = '/root/UnifiedReward-main/UnifiedReward-Think/vllm_eval/output'
    os.makedirs(output_dir, exist_ok=True)
    correct_file_path = os.path.join(output_dir, 'eval_correct.jsonl')
    wrong_file_path = os.path.join(output_dir, 'eval_wrong.jsonl')
    sampling_params_dict = {"max_tokens": 4096}
    prompt_template = "Given a caption and two images generated based on this caption, please analyze in detail the two provided images. Evaluate them on various dimensions such as semantic consistency (how closely the image content aligns with the caption), aesthetics (composition, color usage, artistic expression), authenticity (realism and attention to detail), and any other factors you deem relevant. For each evaluation dimension, provide a score between 1-10 for both images (e.g., Image 1: 8/10, Image 2: 6/10) and provide a concise rationale for the score. Calculate the total score for each image by summing all dimension scores. Use a chain-of-thought process to detail your reasoning steps, and enclose all your detailed reasoning within <think> and </think> tags. Then, in the <answer> tag, output exactly one of the following strings: \'Image 1 is better\' or \'Image 2 is better\' based on the total scores. No additional text is allowed in the <answer> section.\n\nExample output format:\n<think>\n1. Semantic consistency: Image 1 (9/10) - ...; Image 2 (7/10) - ...\n2. Aesthetics: Image 2 (8/10) - ...; Image 1 (8/10) - ...\n3. Authenticity: Image 1 (8/10) - ...; Image 2 (5/10) - ...\n[Additional dimensions if any]: Image 2 (8/10) - ...; Image 1 (6/10) - ...\nTotal score:\nImage 1: 9+8+8+6=31\nImage 2: 7+8+5+8=28\n</think>\n<answer>Image 1 is better</answer>\n**Note: In the example above, scores and the final answer are placeholders meant only to demonstrate the format. Your actual evaluation should be based on the quality of two given images.**\n\nYour task is provided as follows:\nText Caption: [{prompt}]"

    # --- 阶段一：设置数据加载器 ---
    dataset = HPDDataset(json_file_path=data_file_path, base_path=base_path)
    collate_fn_instance = CollateFn(prompt_template)
    dataloader = DataLoader(
        dataset, batch_size=128, shuffle=False, num_workers=6,
        pin_memory=True, collate_fn=collate_fn_instance
    )

    with ThreadPoolExecutor(max_workers=32) as io_executor, \
         ProcessPoolExecutor(max_workers=os.cpu_count() or 8) as cpu_executor, \
         open(correct_file_path, 'w', encoding='utf-8') as f_correct, \
         open(wrong_file_path, 'w', encoding='utf-8') as f_wrong:
        
        async with AsyncOpenAI(base_url=OPENAI_API_BASE, api_key="EMPTY") as client:
            request_queue = Queue(maxsize=1024)
            result_queue = Queue(maxsize=1024)
            loop = asyncio.get_running_loop()

            producer_task = loop.run_in_executor(io_executor, producer, request_queue, dataloader)

            with tqdm(total=len(dataset), desc="  Inferencing", position=0) as pbar_infer, \
                 tqdm(total=len(dataset), desc="Result Writing", position=1) as pbar_result:

                num_inference_workers = 16
                inference_tasks = [
                    asyncio.create_task(inference_worker(
                        client, request_queue, result_queue, loop, io_executor,
                        MODEL_NAME, sampling_params_dict, pbar_infer
                    )) for _ in range(num_inference_workers)
                ]

                num_result_workers = 4
                result_tasks = [
                    asyncio.create_task(result_worker(
                        result_queue, loop, io_executor, cpu_executor, 
                        f_correct, f_wrong, pbar_result
                    )) for _ in range(num_result_workers)
                ]


                # 1. 等待生产者完成所有数据的加载
                await producer_task
                print("Producer finished. All data loaded into request_queue.")
                
                # 2. 向请求队列发送“哨兵”，通知所有推理worker可以结束了
                for _ in range(num_inference_workers):
                    await loop.run_in_executor(io_executor, request_queue.put, None)
                
                # 3. 等待所有推理任务完成
                await asyncio.gather(*inference_tasks)
                print("All inference workers finished.")

                # 4. 推理任务全部结束后，结果队列也不会再有新数据，此时向结果队列发送“哨兵”
                for _ in range(num_result_workers):
                    await loop.run_in_executor(io_executor, result_queue.put, None)

                # 5. 等待所有结果处理任务完成
                await asyncio.gather(*result_tasks)
                print("All result workers finished.")

    print("Pipeline finished gracefully.")


if __name__ == "__main__":
    asyncio.run(main())