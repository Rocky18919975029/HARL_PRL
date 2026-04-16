import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# 挑一个你确定有 eval 数据的实验目录
exp_dir = "results/ablation_study/mamujoco/Humanoid-v2-17x1/happo/self_supervised"

print(f"Scanning directory: {exp_dir}\n")
for root, _, files in os.walk(exp_dir):
    for f in files:
        if "events.out.tfevents" in f:
            event_file = os.path.join(root, f)
            ea = EventAccumulator(event_file)
            ea.Reload()
            tags = ea.Tags().get('scalars', [])
            if tags:
                # 打印所属的文件夹名和它内部的真实 Tag
                folder_name = os.path.basename(root)
                print(f"Folder: {folder_name}")
                print(f" -> Real Tags inside: {tags}\n")