# plot.py

import os
import re
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from scipy.interpolate import interp1d


# =========================
# NeurIPS-style plotting
# =========================

sns.set_theme(style="whitegrid")

PLOT_COLORS = [
    "#4C72B0",  # blue
    "#55A868",  # green
    "#C44E52",  # red
    "#8172B3",  # purple
    "#CCB974",  # yellow-brown
    "#64B5CD",  # cyan
    "#222222",  # near black
    "#937860",  # brown
    "#DA8BC3",  # pink
    "#8C8C8C",  # gray
]

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.figsize": (6.5, 4.2),
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.6,
})


def format_legend_name(exp_name):
    """
    Format experiment name for clean paper-style legend.

    Rules:
        PRL_ours / HAPPO_PRL / xxx_PRL_xxx -> PRL
        HAPPO_baseline -> HAPPO
        MAPPO_baseline -> MAPPO
    """

    name = exp_name
    if name.endswith("_baseline"):
        name = name[:-len("_baseline")]

    return name.replace("_", " ")



def infer_title_from_save_dir(save_dir):
    """
    Use the last directory name as figure title.
    Example:
        ./plots/grf_counterattack_hard -> grf_counterattack_hard
    """
    norm_path = os.path.normpath(save_dir)
    return os.path.basename(norm_path)


def smooth_curve(scalars, weight=0.85):
    """
    Exponential moving average smoothing.
    """
    if len(scalars) == 0:
        return scalars

    last = scalars[0]
    smoothed = []

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
    Traverse multiple directories and collect experiment / seed / timestamp paths.
    """
    exp_dict = defaultdict(lambda: defaultdict(list))

    pattern = re.compile(
        r"(seed-\d+)-(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})"
    )

    for root_dir in root_dirs:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            dirname = os.path.basename(dirpath)
            match = pattern.match(dirname)

            if match:
                seed_str = match.group(1)
                timestamp_str = match.group(2)

                exp_name = os.path.basename(os.path.dirname(dirpath))

                if include_list is not None and exp_name not in include_list:
                    continue

                tb_folder = os.path.join(dirpath, "logs")
                if not os.path.exists(tb_folder):
                    tb_folder = dirpath

                unique_seed_id = f"{root_dir}_{seed_str}"

                exp_dict[exp_name][unique_seed_id].append(
                    (timestamp_str, tb_folder)
                )

    for exp_name in exp_dict:
        for seed_id in exp_dict[exp_name]:
            exp_dict[exp_name][seed_id].sort(key=lambda x: x[0])
            exp_dict[exp_name][seed_id] = [
                x[1] for x in exp_dict[exp_name][seed_id]
            ]

    return exp_dict


def read_tb_logs(folder, tags):
    """
    Read specified TensorBoard scalar tags from one log folder and its subfolders.
    """
    data = {tag: {} for tag in tags}
    found_any_tags = set()

    for root_path, dirs, files in os.walk(folder):
        if any(f.startswith("events.out.tfevents") for f in files):
            print(f"Reading TensorBoard logs from: {root_path}", flush=True)

            try:
                ea = EventAccumulator(root_path, size_guidance={"scalars": 0})
                ea.Reload()

                available_scalars = ea.Tags().get("scalars", [])
                found_any_tags.update(available_scalars)

                for tag in tags:
                    if tag in available_scalars:
                        for event in ea.Scalars(tag):
                            data[tag][event.step] = event.value

            except Exception as e:
                print(f"Error reading {root_path}: {e}")

    if not hasattr(read_tb_logs, "printed_tags") and found_any_tags:
        print(
            "\n  [Debug] Found the following scalars across all event files:\n"
            f"  {list(found_any_tags)}\n"
        )
        read_tb_logs.printed_tags = True

    return data


def maybe_downsample_qmix(common_steps, values, exp_name, qmix_stride):
    """
    Downsample QMIX curve if qmix_stride > 1.
    Only affects plotting density.
    """
    if qmix_stride <= 1:
        return common_steps, values

    if "QMIX" not in exp_name.upper():
        return common_steps, values

    return common_steps[::qmix_stride], values[::qmix_stride]


def build_exp_order_and_color_map(exp_dict, include_list=None):
    """
    Build deterministic experiment order and color mapping.

    If include_list is provided:
        colors strictly follow include_list order.
    Otherwise:
        colors follow sorted experiment names.
    """
    if include_list is not None:
        exp_order = [exp_name for exp_name in include_list if exp_name in exp_dict]
    else:
        exp_order = sorted(exp_dict.keys())

    if len(exp_order) > len(PLOT_COLORS):
        raise ValueError(
            f"Number of experiments ({len(exp_order)}) exceeds number of preset colors "
            f"({len(PLOT_COLORS)}). Please expand PLOT_COLORS."
        )

    color_map = {
        exp_name: PLOT_COLORS[i]
        for i, exp_name in enumerate(exp_order)
    }

    return exp_order, color_map


def add_ordered_legend(ax, exp_order):
    """
    Force legend order to follow exp_order.
    Only shows methods that actually appear in the current figure.
    """
    handles, labels = ax.get_legend_handles_labels()

    label_to_handle = {}
    for handle, label in zip(handles, labels):
        if label not in label_to_handle:
            label_to_handle[label] = handle

    ordered_handles = []
    ordered_labels = []

    for exp_name in exp_order:
        label_name = format_legend_name(exp_name)
        if label_name in label_to_handle and label_name not in ordered_labels:
            ordered_handles.append(label_to_handle[label_name])
            ordered_labels.append(label_name)

    if ordered_handles:
        ax.legend(
            ordered_handles,
            ordered_labels,
            loc="best",
            frameon=True,
            fancybox=False,
            framealpha=0.9,
            edgecolor="0.8",
            borderpad=0.4,
            handlelength=2.2,
        )


def plot_experiments(
    exp_dict,
    target_tags,
    save_dir,
    smooth_weight,
    qmix_stride=1,
    include_list=None,
    max_step=None,
):
    """
    Plot multi-seed curves with mean and std shading.
    """
    os.makedirs(save_dir, exist_ok=True)

    exp_order, color_map = build_exp_order_and_color_map(
        exp_dict,
        include_list=include_list,
    )
    figure_title = infer_title_from_save_dir(save_dir)

    print("\n🎨 Fixed color mapping:")
    for exp_name in exp_order:
        print(f"  - {exp_name} -> {color_map[exp_name]}")

    for tag in target_tags:
        fig, ax = plt.subplots()
        has_data_for_tag = False

        for exp_name in exp_order:
            seeds_dict = exp_dict[exp_name]

            all_seed_steps = []
            all_seed_values = []

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
            
            if max_step is not None:
                max_step = min(max_step, max([s[-1] for s in all_seed_steps]))

            if min_step >= max_step:
                print(
                    f"⚠️ Skipping {exp_name} / {tag}: "
                    f"min_step={min_step}, max_step={max_step}"
                )
                continue

            common_steps = np.linspace(min_step, max_step, 500)

            interpolated_values = []

            for steps, values in zip(all_seed_steps, all_seed_values):
                steps, idx = np.unique(steps, return_index=True)
                values = np.array(values)[idx]

                if len(steps) > 1:
                    f = interp1d(
                        steps,
                        values,
                        bounds_error=False,
                        fill_value=np.nan,
                    )
                    interpolated_values.append(f(common_steps))

            if not interpolated_values:
                continue

            interpolated_values = np.array(interpolated_values)

            mean_vals = np.nanmean(interpolated_values, axis=0)
            std_vals = np.nanstd(interpolated_values, axis=0)

            mean_vals_smooth = smooth_curve(mean_vals, weight=smooth_weight)

            plot_steps, plot_mean = maybe_downsample_qmix(
                common_steps,
                mean_vals_smooth,
                exp_name,
                qmix_stride,
            )

            _, plot_std = maybe_downsample_qmix(
                common_steps,
                std_vals,
                exp_name,
                qmix_stride,
            )

            label_name = format_legend_name(exp_name)
            color = color_map[exp_name]

            ax.plot(
                plot_steps,
                plot_mean,
                label=label_name,
                linewidth=1.6,
                color=color,
            )

            if len(all_seed_steps) > 1:
                ax.fill_between(
                    plot_steps,
                    plot_mean - plot_std,
                    plot_mean + plot_std,
                    alpha=0.12,
                    color=color,
                    linewidth=0,
                )

        if has_data_for_tag:
            ax.set_title(figure_title, pad=8, fontweight="semibold")
            ax.set_xlabel("Environment Steps")
            ax.set_ylabel(tag.replace("_", " "))

            ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

            add_ordered_legend(ax, exp_order)

            ax.spines["left"].set_linewidth(0.8)
            ax.spines["bottom"].set_linewidth(0.8)
            ax.grid(True, alpha=0.22, linewidth=0.5)

            plt.tight_layout()

            safe_tag_name = tag.replace("/", "_")
            png_path = os.path.join(save_dir, f"{safe_tag_name}.png")
            pdf_path = os.path.join(save_dir, f"{safe_tag_name}.pdf")

            plt.savefig(png_path, bbox_inches="tight")
            plt.savefig(pdf_path, bbox_inches="tight")

            print(f"✅ Successfully plotted and saved: {png_path}")
            print(f"✅ Successfully plotted and saved: {pdf_path}")
        else:
            print(f"⚠️ Tag '{tag}' not found in any experiments. Skipped.")

        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot HARL results from TensorBoard logs with multi-seed shading."
    )

    parser.add_argument(
        "--dirs",
        type=str,
        nargs="+",
        required=True,
        help=(
            "List of root directories or specific experiment directories, "
            "e.g. results/seed_1/happo/HAPPO_baseline"
        ),
    )

    parser.add_argument(
        "--include",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional: only plot these specific experiment names, "
            "e.g. HAPPO_baseline PRL_ours"
        ),
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="./plots",
        help="Directory to save plots. Defaults to ./plots.",
    )

    parser.add_argument(
        "--smooth",
        type=float,
        default=0.90,
        help="EMA smoothing factor from 0.0 to 1.0. Default is 0.85.",
    )

    parser.add_argument(
        "--qmix_stride",
        type=int,
        default=1,
        help=(
            "Downsample QMIX curves by taking every N-th point. "
            "Only applies to experiment names containing QMIX."
        ),
    )

    parser.add_argument(
        "--max_step",
        type=float,
        default=None,
        help="Optional maximum environment step to plot.",
    )

    args = parser.parse_args()

    target_tags = [
        "eval_average_episode_rewards",
        "train_episode_rewards",
        "eval_win_rate",
        "eval_score_rate",
    ]

    print(f"🔍 Scanning {len(args.dirs)} directory paths...")

    if args.include:
        print(f"🎯 Filtering active. Only keeping experiments named: {args.include}")

    exp_dict = get_experiment_groups(
        args.dirs,
        include_list=args.include,
    )

    if not exp_dict:
        print(
            "❌ No valid experiment logs found matching your criteria. "
            "Please check your paths or --include names."
        )
        exit(1)

    print(f"\n📊 Found {len(exp_dict)} experiments ready to plot:")
    for exp_name, seeds in exp_dict.items():
        print(f"  - {exp_name}: {len(seeds)} total runs/seeds")

    print("\n🚀 Starting data extraction and plotting...")

    plot_experiments(
    exp_dict=exp_dict,
    target_tags=target_tags,
    save_dir=args.save_dir,
    smooth_weight=args.smooth,
    qmix_stride=args.qmix_stride,
    include_list=args.include,
    max_step=args.max_step,
    )

    print("🎉 All done!")