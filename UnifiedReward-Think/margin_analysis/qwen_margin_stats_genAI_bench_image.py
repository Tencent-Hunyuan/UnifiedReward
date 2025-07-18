from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from datasets import load_from_disk, load_dataset
import pandas as pd
from PIL import Image
import torch
import tqdm
import os
import random
import json
import re

############# UTILS #############
def extract_score_diff(output_text: str) -> int:
    sums = re.findall(r'Image\s*[12]:.*?=(\d+)', output_text)
    if len(sums) != 2:
        return None
    score1, score2 = map(int, sums)
    return score1 - score2


############# INSTRUCTION #############
INSTRUCTION = """Given a caption and two images generated based on this caption, please evaluate them on three dimensions: semantic consistency (how closely the image content aligns with the caption), aesthetics (composition, color usage, artistic expression), authenticity (realism and attention to detail). 
For each evaluation dimension, provide a score between 1-10 for both images (e.g., Image 1: 8/10, Image 2: 6/10). Calculate the total score for each image by summing all dimension scores. Enclose these steps within <think> and </think> tags. You should only provide scores, no additional text is allowed in the <think> section.
Then, in the <answer> tag, output exactly one of the following strings: \'Image 1 is better\' or \'Image 2 is better\' based on the total scores. No additional text is allowed in the <answer> section.\n\n
Example output format:\n
<think>
SC: 9, 7\n
Aesthetics: 8, 8\n
Authenticity: 9, 5\n
Image 1: 9+8+8=25\n
Image 2: 7+8+5=20\n
</think>\n
<answer>Image 1 is better</answer>\n
**Note: In the example above, scores and the final answer are placeholders meant only to demonstrate the format. Your actual evaluation should be based on the quality of two given images.**\n\n
Your task is provided as follows:\n
Text Caption: [{prompt}]
"""

############# START #############

model_path = 'CodeGoat24/UnifiedReward-Think-qwen-7b'
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", device_map={"": 'cuda:0'}
)
processor = AutoProcessor.from_pretrained(model_path)


dataset = load_dataset("TIGER-Lab/GenAI-Bench", 'image_generation')['test']

correct = 0
correct_tie = 0
num_all = 0
num_all_tie = 0

log_dir = os.getenv("LOG_DIR", "logs/margin")
os.makedirs(log_dir, exist_ok=True)
save_freq = int(os.getenv("LOG_SAVE_FREQ", 25))

results = []
for i in tqdm.trange(len(dataset)):
    data = dataset[i]

    if 'both' in data['vote_type'] or 'tie' in data['vote_type']:
        num_all_tie += 1
        num_all += 1
        continue

    if random.choices([True, False])[0]:
        left_image = data['right_image'].resize((512, 512))
        right_image = data['left_image'].resize((512, 512))
        if 'left' in data['vote_type']:
            data['vote_type'] = 'right'
        elif 'right' in data['vote_type']:
            data['vote_type'] = 'left'
    else:
        left_image = data['left_image'].resize((512, 512))
        right_image = data['right_image'].resize((512, 512))

    prompt = data['prompt']


    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": left_image},
                {"type": "image", "image": right_image},
                {
                    "type": "text",
                    "text": INSTRUCTION.format(prompt=prompt)
                },
            ],
        }
    ]


    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)


    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

 
    if 'left' in data['vote_type']:
        answer = 'Image 1 is better'
    elif 'right' in data['vote_type']:
        answer = 'Image 2 is better'
    else:
        answer = 'Both images are equally good'
        num_all_tie += 1

    num_all += 1
    is_correct = 0
    if answer in output_text:
        is_correct = 1
        correct += 1
        if data['vote_type'] == 'tie':
            correct_tie += 1
    
    print(f"\n-- Sample {i} -- , Expected: {data['vote_type']}")
    print(output_text)

    diff = extract_score_diff(output_text)
    if diff is None:
        print(f"Cannot parse the result found: {output_text}")
        continue

    margin = abs(diff)
    results.append({
        'margin':  margin,
        'correct': is_correct
    })
    print(f"Margin: {margin}, Correct: {is_correct}")

    # accuracy so far
    print(f"Acc.: {correct} / {num_all} = {correct / num_all:.4f}")
    print(f"Acc. w/o tie: ({correct} - {correct_tie}) / ({num_all} - {num_all_tie}) = {(correct - correct_tie) / (num_all - num_all_tie):.4f}")

    if (i+1) % save_freq == 0 or i == len(dataset) - 1:
        df_partial = pd.DataFrame(results)
        
        stats = (
            df_partial
            .groupby('margin')['correct']
            .agg(num_samples='count', accuracy='mean')
            .sort_index()
        )
        
        out_path = os.path.join(log_dir, f"stats_after_{i+1}.txt")
        with open(out_path, 'w') as f:
            f.write(stats.to_string())
        print(f"Saved stats snapshot {i+1} to {out_path}")


# final accuracy
print(f"Acc.: {correct} / {num_all} = {correct / num_all:.4f}")
print(f"Acc. w/o tie: ({correct} - {correct_tie}) / ({num_all} - {num_all_tie}) = {(correct - correct_tie) / (num_all - num_all_tie):.4f}")
