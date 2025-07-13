"""
convert_yaml_to_dataset_info.py

# train_data.yaml -> dataset_info.json
# 将 UnifiedReward 训练数据集转换为 LlamaFactory 可用的格式
"""

import json
import pathlib
import re
import uuid

import yaml

YAML_FILE = "train_data.yaml"  # 你的 YAML
ROOT = pathlib.Path(".")  #
OUT = ROOT / "dataset_info.json"

info = {}
for item in yaml.safe_load(open(YAML_FILE))["datasets"]:
    # 生成唯一昵称，例如 ur_llava_critic_pairwise
    nick = (
        "ur_"
        + re.sub(
            r"[^\w]", "_", pathlib.Path(item["json_path"]).parent.as_posix()
        ).lower()
    )
    if nick in info:
        nick += "_" + uuid.uuid4().hex[:4]

    pairwise = any(k in item["json_path"].lower() for k in ["pairwise", "dpo"])
    is_video = any(
        k in item.get("image_folder", "").lower() for k in ["video", "frame"]
    )

    # 去除 dataset/ 前缀，加上 _abs 后缀
    p = pathlib.Path(item["json_path"])                   # 例如 dataset/HPD/train_data.json
    rel = p.relative_to("dataset")                # => HPD/train_data.json （Path 对象）
    new_path = rel.with_stem(rel.stem + "_abs")   # => HPD/train_data_abs.json
    file_name = str(new_path)                     # 转成字符串写进 dataset_info.json
    print(file_name)

    # sampling strategy
    sampling = False
    if "sampling_strategy" in item:
        sampling = False if item["sampling_strategy"] == "all" else True
        num_samples = item["sampling_strategy"].split(":")[-1]

    info[nick] = {
        "file_name": file_name,
        "formatting": "sharegpt",
        # **({"ranking": True} if pairwise else {}),
        "columns": {
            "messages": "conversations",
            **({"chosen": "chosen", "rejected": "rejected"} if pairwise else {}),
            **({"videos": "videos"} if is_video else {"images": "images"}),
        },
    }
    if sampling:
        info[nick]["num_samples"] = int(num_samples)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(info, ensure_ascii=False, indent=2))
print(f"已生成 {len(info)} 条描述 → {OUT}")

