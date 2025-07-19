import json

# HPD
j = 'dataset/HPD/train_data.json'
with open(j, "r", encoding="utf-8") as f:
    data = json.load(f)

target_field = "images"

for item in data:
    for i in range(len(item[target_field])):
        item[target_field][i] = j[31:j.rfind('/')+1] + item[target_field][i]

with open(j, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

# LiFT-HRA
j = 'dataset/LiFT-HRA/train_data.json'
with open(j, "r", encoding="utf-8") as f:
    data = json.load(f)

target_field = "video"

for item in data:
    item[target_field] = j[31:j.rfind('/')+1] + item[target_field]

with open(j, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

# LLaVA-Critic/pairwise
j = 'dataset/LLaVA-Critic/pairwise/train_data.json'
with open(j, "r", encoding="utf-8") as f:
    data = json.load(f)

target_field = "image"

for item in data:
    item[target_field] = j[31:j.rfind('/')+1] + item[target_field]

with open(j, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

# LLaVA-Critic/pointwise
j = 'dataset/LLaVA-Critic/pointwise/train_data.json'
with open(j, "r", encoding="utf-8") as f:
    data = json.load(f)

target_field = "image"

for item in data:
    item[target_field] = j[31:j.rfind('/')+1] + item[target_field]

with open(j, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

# OIP
j = 'dataset/OIP/train_data.json'
with open(j, "r", encoding="utf-8") as f:
    data = json.load(f)

target_field = "images"

for item in data:
    for i in range(len(item[target_field])):
        item[target_field][i] = j[31:j.rfind('/')+1] + item[target_field][i]

with open(j, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

# VideoFeedback
j = 'dataset/VideoFeedback/train_data.json'
with open(j, "r", encoding="utf-8") as f:
    data = json.load(f)

target_field = "images"

for item in data:
    for i in range(len(item[target_field])):
        # item[target_field][i] = j[31:j.rfind('/')+1] + 'frames/' + item[target_field][i]
        item[target_field][i] = item[target_field][i][:20] + '/' + item[target_field][i][20:]

with open(j, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

# VideoDPO
j = 'dataset/VideoDPO/train_data.json'
with open(j, "r", encoding="utf-8") as f:
    data = json.load(f)

target_field = "images"

for item in data:
    for i in range(len(item[target_field])):
        item[target_field][i] = j[31:j.rfind('/')+1] + item[target_field][i]

with open(j, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

# EvalMuse/pairwise
j = 'dataset/EvalMuse/pairwise/train_data.json'
with open(j, "r", encoding="utf-8") as f:
    data = json.load(f)

target_field = "images"

for item in data:
    for i in range(len(item[target_field])):
        item[target_field][i] = j[31:j.rfind('/')-8] + 'images/' + item[target_field][i]

with open(j, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

# EvalMuse/pointwise
j = 'dataset/EvalMuse/pointwise/train_data.json'
with open(j, "r", encoding="utf-8") as f:
    data = json.load(f)

target_field = "image"

for item in data:
    item[target_field] = j[31:j.rfind('/')-9] + 'images/' + item[target_field]

with open(j, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

# ShareGPTVideo-DPO/pairwise
j = 'dataset/ShareGPTVideo-DPO/pairwise/train_data.json'
with open(j, "r", encoding="utf-8") as f:
    data = json.load(f)

target_field = "images"

for item in data:
    for i in range(len(item[target_field])):
        item[target_field][i] = j[31:j.rfind('/')-8] + 'videos/' + item[target_field][i]

with open(j, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

# ShareGPTVideo-DPO/pointwise
j = 'dataset/ShareGPTVideo-DPO/pointwise/train_data.json'
with open(j, "r", encoding="utf-8") as f:
    data = json.load(f)

target_field = "images"

for item in data:
    for i in range(len(item[target_field])):
        # item[target_field][i] = j[31:j.rfind('/')+1] + 'videos/' + item[target_field][i]
        item[target_field][i] = item[target_field][i][:17] + item[target_field][i][27:]

with open(j, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)