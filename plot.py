import os
import re
import argparse
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from scipy.interpolate import interp1d

# 设置更美观的绘图风格
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.figsize': (10, 6)})

def smooth_curve(scalars, weight=0.8):
    """
    指数移动平均 (EMA) 平滑算法
    """
    if len(scalars) == 0:
        return scalars
    last = scalars[0]
    smoothed =[]
    for point in scalars:
        if np.isnan(point):
            smoothed.append(point)
            continue
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)

def get_experiment_groups(root_dirs, include_list=None):
    """
    遍历多目录，收集实验、seed和时间戳路径
    """
    exp_dict = defaultdict(lambda: defaultdict(list))
    
    # 匹配类似于 seed-00001-2026-04-20-21-56-45 的文件夹
    pattern = re.compile(r'(seed-\d+)-(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})')
    
    for root_idx, root_dir in enumerate(root_dirs):
        for dirpath, dirnames, filenames in os.walk(root_dir):
            dirname = os.path.basename(dirpath)
            match = pattern.match(dirname)
            
            if match:
                seed_str = match.group(1)
                timestamp_str = match.group(2)
                # 自动提取所在文件夹的名字作为实验名 (如 HAPPO_baseline)
                exp_name = os.path.basename(os.path.dirname(dirpath))
                
                # ==========================================
                # === 【核心升级】：根据 include 列表进行过滤 ===
                # ==========================================
                if include_list is not None and exp_name not in include_list:
                    continue
                
                tb_folder = os.path.join(dirpath, "logs")
                if not os.path.exists(tb_folder):
                    tb_folder = dirpath # fallback
                    
                # 生成全局唯一的 Seed ID，防止不同目录使用相同的 seed_str 导致冲突
                unique_seed_id = f"{root_dir}_{seed_str}"
                
                exp_dict[exp_name][unique_seed_id].append((timestamp_str, tb_folder))
            
    # 排序并提纯路径（处理断点重续）
    for exp_name in exp_dict:
        for seed_id in exp_dict[exp_name]:
            exp_dict[exp_name][seed_id].sort(key=lambda x: x[0])
            exp_dict[exp_name][seed_id] = [x[1] for x in exp_dict[exp_name][seed_id]]
            
    return exp_dict

def read_tb_logs(folder, tags):
    """
    读取单个 TensorBoard 日志文件夹及其所有子文件夹里的指定 tags
    """
    data = {tag: {} for tag in tags}
    found_any_tags = set()
    
    for root_path, dirs, files in os.walk(folder):
        if any(f.startswith('events.out.tfevents') for f in files):
            try:
                ea = EventAccumulator(root_path, size_guidance={'scalars': 0})
                ea.Reload()
                available_scalars = ea.Tags().get('scalars',[])
                found_any_tags.update(available_scalars)
                
                for tag in tags:
                    if tag in available_scalars:
                        for event in ea.Scalars(tag):
                            data[tag][event.step] = event.value
            except Exception as e:
                print(f"Error reading {root_path}: {e}")
                
    if not hasattr(read_tb_logs, "printed_tags") and found_any_tags:
        print(f"\n  [Debug] Found the following scalars across all event files:\n  {list(found_any_tags)}\n")
        read_tb_logs.printed_tags = True
        
    return data

def plot_experiments(exp_dict, target_tags, save_dir, smooth_weight):
    """
    处理数据并画出带有多 Seed 阴影的图表
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for tag in target_tags:
        plt.figure()
        has_data_for_tag = False
        
        for exp_name, seeds_dict in exp_dict.items():
            all_seed_steps =[]
            all_seed_values =[]
            
            for seed_id, folders in seeds_dict.items():
                merged_data = {}
                for folder in folders:
                    data = read_tb_logs(folder, [tag])
                    merged_data.update(data[tag])
                
                if not merged_data:
                    continue
                    
                steps, values = zip(*sorted(merged_data.items()))
                all_seed_steps.append(steps)
                all_seed_values.append(values)
            
            if not all_seed_steps:
                continue
            
            has_data_for_tag = True
            
            min_step = max([s[0] for s in all_seed_steps])
            max_step = min([s[-1] for s in all_seed_steps])
            
            if min_step >= max_step:
                min_step = min([s[0] for s in all_seed_steps])
                max_step = max([s[-1] for s in all_seed_steps])
                
            common_steps = np.linspace(min_step, max_step, 500)
            
            interpolated_values =[]
            for steps, values in zip(all_seed_steps, all_seed_values):
                steps, idx = np.unique(steps, return_index=True)
                values = np.array(values)[idx]
                
                if len(steps) > 1:
                    f = interp1d(steps, values, bounds_error=False, fill_value=np.nan)
                    interpolated_values.append(f(common_steps))
                    
            if not interpolated_values:
                continue
                
            interpolated_values = np.array(interpolated_values)
            
            # 计算跨多 Seed 的 Mean 和 Std
            mean_vals = np.nanmean(interpolated_values, axis=0)
            std_vals = np.nanstd(interpolated_values, axis=0)
            
            mean_vals_smooth = smooth_curve(mean_vals, weight=smooth_weight)
            
            # 画主曲线
            p = sns.lineplot(x=common_steps, y=mean_vals_smooth, label=exp_name, linewidth=2)
            
            # 画阴影
            if len(all_seed_steps) > 1:
                plt.fill_between(common_steps, 
                                 mean_vals_smooth - std_vals, 
                                 mean_vals_smooth + std_vals, 
                                 alpha=0.15, color=p.lines[-1].get_color())
            
        if has_data_for_tag:
            plt.title(tag.replace("_", " ").title(), fontsize=16, fontweight='bold')
            plt.xlabel("Environment Steps", fontsize=14)
            plt.ylabel(tag, fontsize=14)
            plt.legend(loc='best', frameon=True, fontsize=12)
            plt.tight_layout()
            
            safe_tag_name = tag.replace("/", "_")
            save_path = os.path.join(save_dir, f"{safe_tag_name}.png")
            plt.savefig(save_path, dpi=300)
            print(f"✅ Successfully plotted and saved: {save_path}")
        else:
            print(f"⚠️ Tag '{tag}' not found in any of the experiments. Skipped.")
            
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot HARL results from TensorBoard logs with Multi-Seed shading.")
    
    parser.add_argument("--dirs", type=str, nargs='+', required=True, 
                        help="List of root directories OR specific experiment directories (e.g., results/seed_1/happo/HAPPO_baseline)")
    # 【新增功能】：指定白名单
    parser.add_argument("--include", type=str, nargs='+', default=None,
                        help="Optional: Only plot these specific experiment names (e.g., HAPPO_baseline PRL_ours)")
    
    parser.add_argument("--save_dir", type=str, default="./plots",
                        help="Directory to save the plots. Defaults to './plots' in current directory.")
    parser.add_argument("--smooth", type=float, default=0,
                        help="EMA smoothing factor (0.0 to 1.0). Default is 0.")
    args = parser.parse_args()

    target_tags =[
        "eval_average_episode_rewards",
        "train_episode_rewards",
        "eval_win_rate",
        "eval_score_rate"
    ]
    
    save_dir = args.save_dir
    
    print(f"🔍 Scanning {len(args.dirs)} directory paths...")
    if args.include:
        print(f"🎯 Filtering active! Only keeping experiments named: {args.include}")
        
    exp_dict = get_experiment_groups(args.dirs, include_list=args.include)
    
    if not exp_dict:
        print("❌ No valid experiment logs found matching your criteria. Please check your paths or --include names.")
        exit(1)
        
    print(f"\n📊 Found {len(exp_dict)} experiments ready to plot:")
    for exp_name, seeds in exp_dict.items():
        print(f"  - {exp_name}: {len(seeds)} total runs/seeds")
        
    print("\n🚀 Starting data extraction and plotting...")
    plot_experiments(exp_dict, target_tags, save_dir, args.smooth)
    print("🎉 All done!")