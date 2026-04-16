import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAGS_OF_INTEREST = {
    "Train Reward": ["average_episode_rewards", "train_episode_rewards"],
    "Eval Reward": ["eval_average_episode_rewards", "eval_episode_rewards", "eval_reward"],
    "Aux Loss": ["avg_aux_loss", "avg_action_pred_loss"]
}

COLORS = ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F", "#8491B4", "#91D1C2", "#DC0000", "#7E6148", "#FFB90F", "#9467BD", "#2CA02C"]
LINE_STYLES = ['-', '-', '-', '-', '-', '-', '--', '--', '--', '--', '--', '--']

def set_academic_style():
    sns.set_theme(style="ticks")
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.labelsize": 14, "font.size": 12, "legend.fontsize": 11,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "axes.linewidth": 1.2,
        "grid.linewidth": 0.8, "grid.linestyle": "--", "grid.alpha": 0.6,
        "figure.figsize": (8, 5.5), "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight"
    })

def smooth(scalars, weight=0.85):
    if len(scalars) == 0: return scalars
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)

def extract_data_from_tb(event_file, target_tags):
    """从单个 event file 中提取指定 Tag 的数据"""
    ea = EventAccumulator(event_file, size_guidance={'scalars': 0})
    ea.Reload()
    data = {}
    
    for tag in ea.Tags().get('scalars', []):
        matched_category = None
        tag_lower = tag.lower()
        
        # === 【核心修复：精准阻断子串冲突】 ===
        if "eval" in tag_lower and "reward" in tag_lower:
            matched_category = "Eval Reward"
        elif "reward" in tag_lower and "eval" not in tag_lower:  # 明确排除 eval
            matched_category = "Train Reward"
        elif "aux_loss" in tag_lower or "action_pred_loss" in tag_lower:
            matched_category = "Aux Loss"
            
        if matched_category:
            events = ea.Scalars(tag)
            if matched_category not in data:
                data[matched_category] = {}
            for e in events:
                data[matched_category][e.step] = e.value
                
    return data

def process_single_experiment(exp_dir):
    print(f"Processing Experiment: {exp_dir}")
    experiment_data = {} 
    raw_seed_data = {} 

    # 直接遍历该实验目录下的所有文件，挖地三尺找 event 文件
    for root, _, files in os.walk(exp_dir):
        for f in files:
            if "events.out.tfevents" in f:
                event_file = os.path.join(root, f)
                match = re.search(r'(seed-\d+)', root)
                if not match: continue
                seed_name = match.group(1)
                
                try:
                    run_data = extract_data_from_tb(event_file, TAGS_OF_INTEREST)
                    if any(len(v) > 0 for v in run_data.values()):
                        if seed_name not in raw_seed_data:
                            raw_seed_data[seed_name] = []
                        raw_seed_data[seed_name].append((event_file, run_data))
                except Exception as e:
                    print(f"Warning: Failed to read {event_file}. Error: {e}")

    for seed_name, files_and_data in raw_seed_data.items():
        files_and_data.sort(key=lambda x: x[0])
        merged_seed_data = {} 
        for _, run_data in files_and_data:
            for tag, step_val_dict in run_data.items():
                if tag not in merged_seed_data:
                    merged_seed_data[tag] = {}
                merged_seed_data[tag].update(step_val_dict)
        
        for tag, step_val_dict in merged_seed_data.items():
            if not step_val_dict: continue
            steps = np.array(sorted(step_val_dict.keys()))
            values = np.array([step_val_dict[s] for s in steps])
            if tag not in experiment_data:
                experiment_data[tag] = []
            experiment_data[tag].append((steps, values))
            
    return experiment_data

def plot_experiments(exp_dirs, exp_labels=None, smooth_weight=0.85):
    set_academic_style()
    if exp_labels is None:
        exp_labels = [os.path.basename(os.path.normpath(d)) for d in exp_dirs]
        
    all_data = {tag: {} for tag in TAGS_OF_INTEREST.keys()}
    
    for idx, exp_dir in enumerate(exp_dirs):
        exp_data = process_single_experiment(exp_dir)
        print(f"  [{exp_labels[idx]}] Valid data tags collected: {list(exp_data.keys())}")
        for tag, seed_list in exp_data.items():
            all_data[tag][exp_labels[idx]] = seed_list

    for tag in TAGS_OF_INTEREST.keys():
        tag_data = all_data[tag]
        if not tag_data:
            print(f"--- Skipping '{tag}': No data points found in any experiment (Maybe training stopped before first eval?). ---")
            continue
            
        is_all_zero = True
        for exp_name, seed_list in tag_data.items():
            for steps, values in seed_list:
                if np.any(np.abs(values) > 1e-6):
                    is_all_zero = False; break
        if tag == "Aux Loss" and is_all_zero:
            continue
            
        fig, ax = plt.subplots()
        plot_count = 0
        
        for idx, (exp_name, seed_list) in enumerate(tag_data.items()):
            if not seed_list: continue
            color = COLORS[idx % len(COLORS)]
            linestyle = LINE_STYLES[idx % len(LINE_STYLES)]
            
            min_step = max([s[0] for s, v in seed_list])
            max_step = min([s[-1] for s, v in seed_list])
            
            if min_step > max_step: continue
            if min_step == max_step:
                print(f"  Warning: {exp_name} '{tag}' only has 1 valid data point! Cannot draw a line.")
                continue
                
            common_steps = np.linspace(min_step, max_step, 500)
            interpolated_values = []
            for steps, values in seed_list:
                if len(steps) < 2: continue
                smoothed_values = smooth(values, weight=smooth_weight)
                f = interp1d(steps, smoothed_values, kind='linear', bounds_error=False, fill_value="extrapolate")
                interpolated_values.append(f(common_steps))
            
            if not interpolated_values: continue
            interpolated_values = np.array(interpolated_values)
            mean_vals = np.mean(interpolated_values, axis=0)
            std_vals = np.std(interpolated_values, axis=0)
            
            ax.plot(common_steps, mean_vals, label=exp_name, color=color, linestyle=linestyle, linewidth=2.0)
            ax.fill_between(common_steps, mean_vals - std_vals, mean_vals + std_vals, color=color, alpha=0.15, edgecolor='none')
            plot_count += 1
            
        if plot_count == 0:
            plt.close(fig)
            continue
            
        ax.set_title(tag, fontsize=16, fontweight='bold', pad=15)
        ax.set_xlabel("Environment Steps", fontsize=14)
        ax.set_ylabel("Value", fontsize=14)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x*1e-6:g}M"))
        ax.grid(True)
        sns.despine(ax=ax)
        
        num_items = plot_count
        ncols = 2 if num_items <= 6 else (3 if num_items <= 9 else 4)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=ncols, frameon=False, prop={'size': 11})
        
        save_dir = "humanoid_visualization"
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, tag.replace(" ", "_").lower() + ".pdf")
        plt.savefig(filename)
        plt.savefig(filename.replace(".pdf", ".png"))
        print(f"Saved plot -> {filename}")
        plt.close(fig)

if __name__ == "__main__":
    experiments = [
        "/home/zeshenghong/HARL/results/mamujoco/Humanoid-v2-17x1/mappo/harl_mappo",
        "/home/zeshenghong/HARL/results/mamujoco/Humanoid-v2-17x1/happo/harl_happo",
        "results/ablation_study/mamujoco/Humanoid-v2-17x1/happo/self_supervised_conditional_policy",
        "results/ablation_study/mamujoco/Humanoid-v2-17x1/happo/self_supervised_conditional_policy_rnn",
        "results/ablation_study/mamujoco/Humanoid-v2-17x1/happo/happo_deepNet",
        "results/ablation_study/mamujoco/Humanoid-v2-17x1/happo/happo_deepNet_rnn",
        "results/ablation_study/mamujoco/Humanoid-v2-17x1/happo/self_supervised",
        "results/ablation_study/mamujoco/Humanoid-v2-17x1/happo/self_supervised_rnn",
        "results/ablation_study/mamujoco/Humanoid-v2-17x1/happo/HAPPO_teammate_modeling_MLP",
        "results/ablation_study/mamujoco/Humanoid-v2-17x1/happo/HAPPO_teammate_modeling_RNN",
    ]
    
    labels = [
        "MAPPO (Baseline)", "HAPPO (Baseline)", "HAPPO + cond Actor (MLP)",
        "HAPPO + cond Actor (RNN)", "HAPPO + Deep Model (MLP)", "HAPPO + Deep Model (RNN)",
        "HAPPO + Rep Learning (MLP)", "HAPPO + Rep Learning (RNN)",
        "HAPPO + Teammate Modeling (MLP)", "HAPPO + Teammate Modeling (RNN)"
    ]
    
    plot_experiments(experiments, exp_labels=labels, smooth_weight=0.85)