# UnifiedReward 使用说明

## 脚本说明

- **build_ur_pipeline.py**  
  构造 `data.json` 的简易实现。  

- **main.sh**  
  对 Geneval 输出的图片和对应 json 进行后处理，并调用 **UnifiedReward** 生成 `data.json`。  

- **train_1.py**  
  可替换 `UnifiedReward/DiffusionDPO/train.py` 作为训练脚本。  

- **dpo.sh**  
  对应的 **DPO 训练启动脚本**。  
