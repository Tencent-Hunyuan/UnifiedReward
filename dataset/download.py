from huggingface_hub import snapshot_download

datasets = [
    "CodeGoat24/ImageGen-CoT-Reward-5K",
    "CodeGoat24/Text-2-Video-Human-Preferences",
    "CodeGoat24/OpenAI-4o_t2i_human_preference",
    "CodeGoat24/LLaVA-Critic-113k",
    "CodeGoat24/OIP",
    "CodeGoat24/ShareGPTVideo-DPO",
    "CodeGoat24/VideoDPO",
    "CodeGoat24/EvalMuse",
    "CodeGoat24/VideoFeedback",
    "CodeGoat24/HPD",
    "CodeGoat24/LiFT-HRA",
]

for repo in datasets:
    snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        local_dir=f"./{repo.split('/')[-1]}",
        resume_download=True,  # 断点续传
        local_dir_use_symlinks=False,
        max_workers=16,
    )

print("所有数据集已下载完成。")

