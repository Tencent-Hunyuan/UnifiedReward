#!/usr/bin/env python
# coding: utf-8
"""
add_dataset_prefix.py

根据 train_data.yaml 中的 json_path 和 image_folder，
给样本里所有 images / videos 短路径加上前缀，并写回新文件。
"""

import json
import pathlib

import yaml

YAML_FILE = "train_data.yaml"  # 你的 YAML
PREFIX_LEN = len("dataset/")  # 用来去掉统一的 dataset/ 前缀（可按需调整）


def add_prefix(record, media_key, prefix):
    """给 record[media_key] 列表里所有媒体路径加前缀"""
    if media_key not in record:
        return

    if isinstance(record[media_key], str):
        # 如果是单个字符串，直接加前缀
        record[media_key] = str(pathlib.Path(prefix) / record[media_key])
    elif isinstance(record[media_key], list):
        # 如果是列表，给每个路径加前缀
        record[media_key] = [str(pathlib.Path(prefix) / p) for p in record[media_key]]


def main():
    cfgs = yaml.safe_load(open(YAML_FILE))["datasets"]
    for cfg in cfgs:
        json_path = pathlib.Path(cfg["json_path"][PREFIX_LEN:])
        media_root = cfg["image_folder"]
        media_root = media_root[PREFIX_LEN:]  # 去掉 dataset/ 前缀，让路径更短

        # 读取原始 JSON
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 判断样本类型：list 或字典（部分 pointwise 可能是 dict 格式）
        if isinstance(data, dict):
            iterable = data.values()
        else:
            iterable = data

        # 如果是 LLaVA-Critic 的数据集，可能没有 images 字段，只有 image 字段，需要转为 images
        if "LLaVA-Critic" in cfg["json_path"]:
            for rec in iterable:
                if "image" in rec and "images" not in rec:
                    rec["images"] = [rec.pop("image")]


        # 根据 keys 批量加前缀
        for rec in iterable:
            add_prefix(rec, "images", media_root)
            add_prefix(rec, "videos", media_root)

        # 写入新文件
        new_path = json_path.with_stem(json_path.stem + "_abs")
        with open(new_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"已写出: {new_path}")


if __name__ == "__main__":
    main()
