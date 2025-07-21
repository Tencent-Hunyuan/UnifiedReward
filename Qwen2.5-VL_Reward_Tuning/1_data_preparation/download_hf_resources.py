import os
import subprocess

# 配置 hf镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 注意：hfd.sh 脚本本身可能不需要 HF_HUB_ENABLE_HF_TRANSFER，但为了保持一致性，可以保留
# os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1' # hfd.sh 脚本可能不依赖这个环境变量

# --- 配置区 ---
# 设置模型下载的本地路径
model_local_dir = "/workspace/model" # 请根据实际环境调整此路径，这是模型最终存放的位置
model_id = "Qwen/Qwen2.5-VL-3B-Instruct" # 要下载的模型ID

# 设置数据集下载的本地路径和ID (如果需要hfd.sh下载整个数据集，请在这里配置并取消hfd_cmd中的相关注释)
# dataset_local_dir = "/work/LLaMA-Factory/data/CodeGoat24" # 数据集最终存放的位置
# dataset_id = "CodeGoat24/ImageGen-CoT-Reward-5K" # 要下载的数据集ID

# 组合出模型在本地的完整路径
local_model_path = os.path.join(model_local_dir, model_id)

# 确保本地目录存在 (hfd.sh 应该会自动创建，但最好确保)
os.makedirs(local_model_path, exist_ok=True)
# os.makedirs(os.path.join(dataset_local_dir, dataset_id), exist_ok=True) # 如果要下载数据集也创建目录

# --- 脚本主逻辑 ---
print("--- Hugging Face 资源自动化下载脚本 ---")
print("本脚本用于自动化下载Hugging Face上的模型和数据集。")

# 1. 下载 hfd.sh 脚本 (如果还没有下载)
hfd_script_path = "./hfd.sh"  # hfd.sh 脚本的保存路径，建议在当前目录
if not os.path.exists(hfd_script_path):
    print("正在下载 hfd.sh 脚本...")
    download_cmd = ["wget", "https://hf-mirror.com/hfd/hfd.sh"]
    try:
        subprocess.run(download_cmd, check=True)
        # 2. 使 hfd.sh 脚本可执行
        print("正在赋予 hfd.sh 脚本执行权限...")
        chmod_cmd = ["chmod", "+x", hfd_script_path]
        subprocess.run(chmod_cmd, check=True)
        print("hfd.sh 脚本下载并已设置为可执行。")
    except subprocess.CalledProcessError as e:
        print(f"错误: 无法下载或设置 hfd.sh 脚本权限: {e}")
        exit(1) # 如果hfd.sh无法下载，则整个脚本无法继续
else:
    print("hfd.sh 脚本已存在。")


# 3. 构建并执行 hfd.sh 命令来下载模型
print(f"\n=== 正在下载模型: {model_id} 到 {local_model_path} ===")

hfd_model_cmd = [
    hfd_script_path,
    model_id,
    "--local-dir",
    local_model_path
]

# 执行 hfd.sh 下载模型
try:
    subprocess.run(hfd_model_cmd, check=True)
    print(f"模型 {model_id} 已成功下载到 {local_model_path}。")
except subprocess.CalledProcessError as e:
    print(f"下载模型 {model_id} 失败: {e}")
    print(f"返回码: {e.returncode}")
    print(f"标准输出: {e.stdout.decode().strip()}")
    print(f"标准错误: {e.stderr.decode().strip()}")
    exit(1) # 模型下载失败则退出

# 4. 构建并执行 hfd.sh 命令来下载数据集 (如果需要)
# 请注意: 本次实验的图片数据集通过 prepare_data.sh 脚本处理，
# 如果ImageGen-CoT-Reward-5K作为一个HF数据集被hfd.sh支持下载，您可以取消注释以下代码块。
# print(f"\n=== 正在下载数据集: {dataset_id} 到 {os.path.join(dataset_local_dir, dataset_id)} ===")
# hfd_dataset_cmd = [
#     hfd_script_path,
#     dataset_id,
#     "--dataset", # 必须加上这个标志表示下载的是数据集
#     "--local-dir",
#     os.path.join(dataset_local_dir, dataset_id)
# ]
# try:
#     subprocess.run(hfd_dataset_cmd, check=True)
#     print(f"数据集 {dataset_id} 已成功下载到 {os.path.join(dataset_local_dir, dataset_id)}。")
# except subprocess.CalledProcessError as e:
#     print(f"下载数据集 {dataset_id} 失败: {e}")
#     print(f"返回码: {e.returncode}")
#     print(f"标准输出: {e.stdout.decode().strip()}")
#     print(f"标准错误: {e.stderr.decode().strip()}")
#     exit(1)


print("\n下载过程结束。")