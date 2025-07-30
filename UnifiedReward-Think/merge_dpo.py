import json
import re

def create_dpo_dataset(correct_file_path, wrong_file_path, output_file_path):
    """
    Compares two .jsonl files with 'conversations' format and creates a DPO dataset.
    For each entry in wrong_file, it searches for a corresponding entry in correct_file.

    Args:
        correct_file_path (str): Path to the jsonl file with correct/chosen responses.
        wrong_file_path (str): Path to the jsonl file with wrong/rejected responses.
        output_file_path (str): Path to save the generated DPO dataset.
    """
    
    # Define the prompt template
    prompt_template = """Given a caption and two images generated based on this caption, please analyze in detail the two provided images. Evaluate them on various dimensions such as semantic consistency (how closely the image content aligns with the caption), aesthetics (composition, color usage, artistic expression), authenticity (realism and attention to detail), and any other factors you deem relevant. For each evaluation dimension, provide a score between 1-10 for both images (e.g., Image 1: 8/10, Image 2: 6/10) and provide a concise rationale for the score. Calculate the total score for each image by summing all dimension scores. Use a chain-of-thought process to detail your reasoning steps, and enclose all your detailed reasoning within <think> and </think> tags. Then, in the <answer> tag, output exactly one of the following strings: 'Image 1 is better' or 'Image 2 is better' based on the total scores. No additional text is allowed in the <answer> section.

Example output format:
<think>
1. Semantic consistency: Image 1 (9/10) - ...; Image 2 (7/10) - ...
2. Aesthetics: Image 2 (8/10) - ...; Image 1 (8/10) - ...
3. Authenticity: Image 1 (8/10) - ...; Image 2 (5/10) - ...
[Additional dimensions if any]: Image 2 (8/10) - ...; Image 1 (6/10) - ...
Total score:
Image 1: 9+8+8+6=31
Image 2: 7+8+5+8=28
</think>
<answer>Image 1 is better</answer>
**Note: In the example above, scores and the final answer are placeholders meant only to demonstrate the format. Your actual evaluation should be based on the quality of two given images.**

Your task is provided as follows:
Text Caption: [{prompt}]"""
    
    def format_problem(original_problem):
        """
        Extract caption from original problem and format it with the template.
        """
        # Remove <image><image> from the original problem to get the caption
        caption = original_problem.replace('<image><image>', '').strip()
        
        # Replace [{prompt}] with the actual caption in the template
        formatted_prompt = prompt_template.replace('[{prompt}]', caption)
        
        # Add <image><image> at the beginning
        return '<image><image>' + formatted_prompt
    
    correct_data = {}
    with open(correct_file_path, 'r', encoding='utf-8') as f_correct:
        for line in f_correct:
            try:
                data = json.loads(line.strip())
                # Handle both conversation format and problem/solution format
                if 'conversations' in data:
                    prompt = next((conv['value'] for conv in data.get('conversations', []) if conv.get('from') == 'human'), None)
                elif 'problem' in data:
                    prompt = data['problem']
                else:
                    continue
                    
                if prompt:
                    correct_data[prompt] = data
            except json.JSONDecodeError as e:
                print(f"Skipping line in correct file due to JSON decode error: {e}")

    with open(wrong_file_path, 'r', encoding='utf-8') as f_wrong, \
         open(output_file_path, 'w', encoding='utf-8') as f_out:

        for line_wrong in f_wrong:
            try:
                data_wrong = json.loads(line_wrong.strip())

                # Handle both conversation format and problem/solution format
                if 'conversations' in data_wrong:
                    prompt_wrong = next((conv['value'] for conv in data_wrong.get('conversations', []) if conv.get('from') == 'human'), None)
                    rejected = next((conv['value'] for conv in data_wrong.get('conversations', []) if conv.get('from') == 'gpt'), None)
                elif 'problem' in data_wrong:
                    prompt_wrong = data_wrong['problem']
                    rejected = data_wrong.get('solution', '')
                else:
                    continue
                
                # Get images from wrong data
                images_wrong = data_wrong.get('images', [])

                matched_prompt_key = None
                for p_correct in correct_data.keys():
                    # Extract caption from both prompts for a more reliable match
                    caption_correct = p_correct.replace('<image><image>', '').strip()
                    caption_wrong = prompt_wrong.replace('<image><image>', '').strip()
                    if caption_correct and caption_wrong and caption_correct == caption_wrong:
                        matched_prompt_key = p_correct
                        break

                if matched_prompt_key:
                    data_correct = correct_data[matched_prompt_key]
                    
                    # Handle both conversation format and problem/solution format for chosen
                    if 'conversations' in data_correct:
                        chosen = next((conv['value'] for conv in data_correct.get('conversations', []) if conv.get('from') == 'gpt'), None)
                    elif 'solution' in data_correct:
                        chosen = data_correct['solution']
                    else:
                        chosen = None
                    
                    # Get images from correct data, fallback to wrong data images if not available
                    images_correct = data_correct.get('images', images_wrong)

                    if prompt_wrong and chosen and rejected:
                        # Format the problem using the new template
                        formatted_problem = format_problem(prompt_wrong)
                        
                        dpo_entry = {
                            'problem': formatted_problem,
                            'chosen': chosen,
                            'rejected': rejected,
                            'images': images_correct  # Add images field
                        }
                        f_out.write(json.dumps(dpo_entry, ensure_ascii=False) + '\n')

            except json.JSONDecodeError as e:
                print(f"Skipping line in wrong file due to JSON decode error: {e}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    correct_file = r'UnifiedReward-Think\vllm_eval\output_think\eval_correct_s3.jsonl'
    wrong_file = r'UnifiedReward-Think\vllm_eval\output\eval_wrong.jsonl'
    output_file = 'dpo_dataset.jsonl'
    create_dpo_dataset(correct_file, wrong_file, output_file)
    print(f"DPO dataset created at: {output_file}")
