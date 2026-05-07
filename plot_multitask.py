# plot_multitask.py

import os
import re
import sys
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


# =========================
# Reporting
# =========================

def add_curve_report(report_events, message):
    if report_events is None:
        print(message)
    else:
        report_events.append(message)


def write_curve_report(save_dir, report_events, report_filename="curve_filter_report.txt"):
    os.makedirs(save_dir, exist_ok=True)
    report_path = os.path.join(save_dir, report_filename)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Curve filtering report\n")
        f.write("======================\n")
        f.write("Command:\n")
        f.write("  " + " ".join(sys.argv) + "\n\n")

        if report_events:
            f.write(f"Total entries: {len(report_events)}\n\n")
            for idx, message in enumerate(report_events, start=1):
                f.write(f"[{idx}] {message}\n")
        else:
            f.write("Total entries: 0\n\n")
            f.write("No missing, unusable, or incomplete curves were excluded.\n")

    print(f"📝 Curve filtering report saved: {report_path} ({len(report_events)} entries)")
    return report_path


# =========================
# Naming / layout helpers
# =========================

def format_legend_name(exp_name):
    """
    Format experiment name for clean paper-style legend.

    Raw PRL variants used in different tasks are treated as the same method:
        PRL_ours / PRL_without_Norm / PRL_lembda_0.5 -> PRL
    """
    if "PRL" in exp_name.upper():
        return "PRL"

    name = exp_name
    if name.endswith("_baseline"):
        name = name[:-len("_baseline")]

    return name.replace("_", " ")


def infer_title_from_save_dir(save_dir):
    norm_path = os.path.normpath(save_dir)
    return os.path.basename(norm_path)


def infer_task_title(task_name):
    return task_name.replace("_", " ")


def parse_layout(layout):
    match = re.match(r"^(\d+)x(\d+)$", layout.lower().strip())
    if not match:
        raise ValueError(
            f"Invalid --layout '{layout}'. Expected format like 2x2, 1x3, or 2x3."
        )
    return int(match.group(1)), int(match.group(2))


def parse_optional_float_list(values, expected_len, arg_name):
    """
    Parse optional per-task numeric lists. Accepts none/None/null/-1 as no limit.
    """
    if values is None:
        return None

    if len(values) != expected_len:
        raise ValueError(
            f"{arg_name} must have the same length as --task_names: "
            f"got {len(values)} values for {expected_len} tasks."
        )

    parsed = []
    for value in values:
        text = str(value).strip()
        if text.lower() in {"none", "null", "na", "n/a", "-1"}:
            parsed.append(None)
        else:
            parsed.append(float(text))
    return parsed


# =========================
# Data reading / grouping
# =========================

def smooth_curve(scalars, weight=0.85):
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

    Same-seed multiple timestamp runs are preserved as a folder list and later merged.
    """
    exp_dict = defaultdict(lambda: defaultdict(list))
    pattern = re.compile(r"(seed-\d+)-(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})")

    for root_dir in root_dirs:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            dirname = os.path.basename(dirpath)
            match = pattern.match(dirname)

            if not match:
                continue

            seed_str = match.group(1)
            timestamp_str = match.group(2)
            exp_name = os.path.basename(os.path.dirname(dirpath))

            if include_list is not None and exp_name not in include_list:
                continue

            tb_folder = os.path.join(dirpath, "logs")
            if not os.path.exists(tb_folder):
                tb_folder = dirpath

            unique_seed_id = f"{root_dir}_{seed_str}"
            exp_dict[exp_name][unique_seed_id].append((timestamp_str, tb_folder))

    for exp_name in exp_dict:
        for seed_id in exp_dict[exp_name]:
            exp_dict[exp_name][seed_id].sort(key=lambda x: x[0])
            exp_dict[exp_name][seed_id] = [x[1] for x in exp_dict[exp_name][seed_id]]

    return exp_dict


def read_tb_logs(folder, tags):
    """Read specified TensorBoard scalar tags from one log folder and its subfolders."""
    data = {tag: {} for tag in tags}
    found_any_tags = set()

    for root_path, dirs, files in os.walk(folder):
        if not any(f.startswith("events.out.tfevents") for f in files):
            continue

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
            "\n  [Debug] Found the following scalars across event files:\n"
            f"  {list(found_any_tags)}\n"
        )
        read_tb_logs.printed_tags = True

    return data


def group_dirs_by_task_name(root_dirs, task_names):
    task_to_dirs = {task_name: [] for task_name in task_names}

    for root_dir in root_dirs:
        norm_dir = os.path.normpath(root_dir)
        for task_name in task_names:
            if task_name in norm_dir:
                task_to_dirs[task_name].append(root_dir)

    return task_to_dirs


def merge_exp_dicts(exp_dicts):
    merged = defaultdict(lambda: defaultdict(list))

    for exp_dict in exp_dicts:
        for exp_name, seeds_dict in exp_dict.items():
            for seed_id, folders in seeds_dict.items():
                merged[exp_name][seed_id].extend(folders)

    return merged


def build_method_order_and_color_map(exp_dict, include_list=None):
    """
    Colors are assigned by canonical display name, not raw experiment name.
    """
    if include_list is not None:
        raw_order = [exp_name for exp_name in include_list if exp_name in exp_dict]
    else:
        raw_order = sorted(exp_dict.keys())

    method_order = []
    for exp_name in raw_order:
        method_name = format_legend_name(exp_name)
        if method_name not in method_order:
            method_order.append(method_name)

    if len(method_order) > len(PLOT_COLORS):
        raise ValueError(
            f"Number of canonical methods ({len(method_order)}) exceeds number of preset colors "
            f"({len(PLOT_COLORS)}). Please expand PLOT_COLORS."
        )

    color_map = {method_name: PLOT_COLORS[i] for i, method_name in enumerate(method_order)}
    return method_order, color_map


def group_exp_dict_by_method(exp_dict, method_order):
    """
    Pool raw experiment names that map to the same canonical method.

    Seed ids are prefixed by raw experiment name so aliases do not overwrite each other.
    """
    method_dict = defaultdict(lambda: defaultdict(list))

    for exp_name, seeds_dict in exp_dict.items():
        method_name = format_legend_name(exp_name)
        if method_name not in method_order:
            continue

        for seed_id, folders in seeds_dict.items():
            unique_seed_id = f"{exp_name}::{seed_id}"
            method_dict[method_name][unique_seed_id].extend(folders)

    return method_dict


# =========================
# Plot helpers
# =========================

def maybe_downsample_qmix(common_steps, values, method_name, qmix_stride):
    if qmix_stride <= 1:
        return common_steps, values
    if "QMIX" not in method_name.upper():
        return common_steps, values
    return common_steps[::qmix_stride], values[::qmix_stride]


def add_ordered_legend(ax, method_order):
    handles, labels = ax.get_legend_handles_labels()
    label_to_handle = {}

    for handle, label in zip(handles, labels):
        if label not in label_to_handle:
            label_to_handle[label] = handle

    ordered_handles = []
    ordered_labels = []
    for label_name in method_order:
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


def collect_ordered_handles_labels(axes, method_order):
    label_to_handle = {}

    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label not in label_to_handle:
                label_to_handle[label] = handle

    ordered_handles = []
    ordered_labels = []
    for label_name in method_order:
        if label_name in label_to_handle and label_name not in ordered_labels:
            ordered_handles.append(label_to_handle[label_name])
            ordered_labels.append(label_name)

    return ordered_handles, ordered_labels


def collect_seed_curves(
    seeds_dict,
    tag,
    method_name=None,
    task_name=None,
    report_missing=True,
    report_events=None,
):
    """Read and merge all TensorBoard events for each seed."""
    seed_curves = []

    for seed_id, folders in seeds_dict.items():
        merged_data = {}

        # Preserve original function: same-seed multiple timestamp runs are merged.
        # Later runs overwrite duplicate steps from earlier runs.
        for folder in folders:
            data = read_tb_logs(folder, [tag])
            merged_data.update(data[tag])

        prefix = f"{task_name} / " if task_name is not None else ""
        method_prefix = f"{method_name} / " if method_name is not None else ""

        if not merged_data:
            if report_missing:
                add_curve_report(
                    report_events,
                    f"Missing curve excluded: "
                    f"{prefix}{method_prefix}{seed_id} / {tag}; "
                    f"reason=no scalar values found for this tag"
                )
            continue

        steps = np.array(sorted(merged_data.keys()), dtype=float)
        values = np.array([merged_data[step] for step in steps], dtype=float)

        valid_mask = np.isfinite(values)
        steps = steps[valid_mask]
        values = values[valid_mask]

        if len(steps) < 2:
            if report_missing:
                add_curve_report(
                    report_events,
                    f"Unusable curve excluded: "
                    f"{prefix}{method_prefix}{seed_id} / {tag}; "
                    f"reason=fewer than 2 finite points"
                )
            continue

        seed_curves.append({
            "seed_id": seed_id,
            "steps": steps,
            "values": values,
            "final_step": float(steps[-1]),
        })

    return seed_curves


def filter_complete_seed_curves(
    seed_curves,
    method_name,
    tag,
    max_step=None,
    task_name=None,
    report_events=None,
):
    """
    Exclude incomplete curves from averaging.

    If max_step is provided, a seed is complete only if final_step >= max_step.
    Otherwise, the longest available seed for task/method/tag defines required_step.
    """
    if not seed_curves:
        return [], None

    if max_step is not None:
        required_step = float(max_step)
    else:
        required_step = max(curve["final_step"] for curve in seed_curves)

    complete_curves = []
    prefix = f"{task_name} / " if task_name is not None else ""

    for curve in seed_curves:
        if curve["final_step"] < required_step:
            add_curve_report(
                report_events,
                f"Incomplete curve excluded: "
                f"{prefix}{method_name} / {curve['seed_id']} / {tag}; "
                f"final_step={curve['final_step']:.0f}, "
                f"required_step={required_step:.0f}"
            )
            continue
        complete_curves.append(curve)

    return complete_curves, required_step


def plot_tag_on_ax(
    ax,
    exp_dict,
    tag,
    method_order,
    color_map,
    smooth_weight,
    qmix_stride=1,
    max_step=None,
    show_xlabel=True,
    show_ylabel=True,
    task_name=None,
    report_events=None,
):
    has_data_for_tag = False
    method_dict = group_exp_dict_by_method(exp_dict, method_order)

    for method_name in method_order:
        if method_name not in method_dict:
            continue

        seed_curves = collect_seed_curves(
            method_dict[method_name],
            tag,
            method_name=method_name,
            task_name=task_name,
            report_missing=True,
            report_events=report_events,
        )

        if not seed_curves:
            prefix = f"{task_name} / " if task_name is not None else ""
            add_curve_report(
                report_events,
                f"Skipping {prefix}{method_name} / {tag}: no usable seed curves found."
            )
            continue

        seed_curves, required_step = filter_complete_seed_curves(
            seed_curves=seed_curves,
            method_name=method_name,
            tag=tag,
            max_step=max_step,
            task_name=task_name,
            report_events=report_events,
        )

        if not seed_curves:
            prefix = f"{task_name} / " if task_name is not None else ""
            add_curve_report(
                report_events,
                f"Skipping {prefix}{method_name} / {tag}: "
                f"no complete seeds remain after incomplete-curve filtering."
            )
            continue

        min_common_step = max(curve["steps"][0] for curve in seed_curves)
        max_common_step = min(curve["steps"][-1] for curve in seed_curves)

        if required_step is not None:
            max_common_step = min(max_common_step, float(required_step))

        if min_common_step >= max_common_step:
            min_common_step = min(curve["steps"][0] for curve in seed_curves)
            max_common_step = max(curve["steps"][-1] for curve in seed_curves)
            if required_step is not None:
                max_common_step = min(max_common_step, float(required_step))

        if min_common_step >= max_common_step:
            prefix = f"{task_name} / " if task_name is not None else ""
            add_curve_report(
                report_events,
                f"Skipping {prefix}{method_name} / {tag}: "
                f"min_step={min_common_step:.0f}, max_step={max_common_step:.0f}"
            )
            continue

        common_steps = np.linspace(min_common_step, max_common_step, 500)
        interpolated_values = []

        for curve in seed_curves:
            steps, idx = np.unique(curve["steps"], return_index=True)
            values = np.array(curve["values"])[idx]
            if len(steps) > 1:
                f = interp1d(steps, values, bounds_error=False, fill_value=np.nan)
                interpolated_values.append(f(common_steps))

        if not interpolated_values:
            continue

        has_data_for_tag = True
        interpolated_values = np.array(interpolated_values)
        mean_vals = np.nanmean(interpolated_values, axis=0)
        std_vals = np.nanstd(interpolated_values, axis=0)
        mean_vals_smooth = smooth_curve(mean_vals, weight=smooth_weight)

        plot_steps, plot_mean = maybe_downsample_qmix(
            common_steps, mean_vals_smooth, method_name, qmix_stride
        )
        _, plot_std = maybe_downsample_qmix(
            common_steps, std_vals, method_name, qmix_stride
        )

        color = color_map[method_name]
        ax.plot(plot_steps, plot_mean, label=method_name, linewidth=1.6, color=color)

        if len(seed_curves) > 1:
            ax.fill_between(
                plot_steps,
                plot_mean - plot_std,
                plot_mean + plot_std,
                alpha=0.12,
                color=color,
                linewidth=0,
            )

    ax.set_xlabel("Environment Steps" if show_xlabel else "")
    ax.set_ylabel(tag.replace("_", " ") if show_ylabel else "")
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.grid(True, alpha=0.22, linewidth=0.5)
    return has_data_for_tag


# =========================
# Plotting functions
# =========================

def plot_experiments(
    exp_dict,
    target_tags,
    save_dir,
    smooth_weight,
    qmix_stride=1,
    include_list=None,
    max_step=None,
    report_events=None,
):
    """Original single-task behavior is preserved."""
    os.makedirs(save_dir, exist_ok=True)

    method_order, color_map = build_method_order_and_color_map(exp_dict, include_list=include_list)
    figure_title = infer_title_from_save_dir(save_dir)

    print("\n🎨 Fixed canonical color mapping:")
    for method_name in method_order:
        print(f"  - {method_name} -> {color_map[method_name]}")

    for tag in target_tags:
        fig, ax = plt.subplots()
        has_data_for_tag = plot_tag_on_ax(
            ax=ax,
            exp_dict=exp_dict,
            tag=tag,
            method_order=method_order,
            color_map=color_map,
            smooth_weight=smooth_weight,
            qmix_stride=qmix_stride,
            max_step=max_step,
            show_xlabel=True,
            show_ylabel=True,
            task_name=None,
            report_events=report_events,
        )

        if has_data_for_tag:
            ax.set_title(figure_title, pad=8, fontweight="semibold")
            add_ordered_legend(ax, method_order)
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


def plot_combined_tasks(
    task_exp_dicts,
    target_tags,
    save_dir,
    smooth_weight,
    layout="2x2",
    qmix_stride=1,
    include_list=None,
    max_step=None,
    task_max_step_map=None,
    report_events=None,
):
    """Plot N tasks into one combined figure for each scalar tag."""
    os.makedirs(save_dir, exist_ok=True)
    rows, cols = parse_layout(layout)
    task_names = list(task_exp_dicts.keys())

    if len(task_names) > rows * cols:
        raise ValueError(
            f"Layout {layout} has only {rows * cols} panels, "
            f"but {len(task_names)} task_names were provided."
        )

    merged_exp_dict = merge_exp_dicts(task_exp_dicts.values())
    method_order, color_map = build_method_order_and_color_map(
        merged_exp_dict, include_list=include_list
    )

    print("\n🎨 Global fixed canonical color mapping:")
    for method_name in method_order:
        print(f"  - {method_name} -> {color_map[method_name]}")

    if task_max_step_map:
        print("\n⏱️ Per-task max steps:")
        for task_name in task_names:
            print(f"  - {task_name}: {task_max_step_map.get(task_name, max_step)}")

    for tag in target_tags:
        fig_width = max(4.6 * cols, 6.5)
        fig_height = max(3.4 * rows, 4.2)
        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
        axes = np.array(axes).reshape(-1)
        has_data_for_tag = False

        for panel_idx, task_name in enumerate(task_names):
            ax = axes[panel_idx]
            row_idx = panel_idx // cols
            task_max_step = task_max_step_map.get(task_name, max_step) if task_max_step_map else max_step

            has_data = plot_tag_on_ax(
                ax=ax,
                exp_dict=task_exp_dicts[task_name],
                tag=tag,
                method_order=method_order,
                color_map=color_map,
                smooth_weight=smooth_weight,
                qmix_stride=qmix_stride,
                max_step=task_max_step,
                show_xlabel=(row_idx == rows - 1),
                show_ylabel=False,
                task_name=task_name,
                report_events=report_events,
            )
            has_data_for_tag = has_data_for_tag or has_data
            ax.set_title(infer_task_title(task_name), pad=8, fontweight="semibold")

        for empty_idx in range(len(task_names), rows * cols):
            axes[empty_idx].axis("off")

        handles, labels = collect_ordered_handles_labels(axes, method_order)
        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.0),
                ncol=min(len(labels), max(1, len(labels))),
                frameon=True,
                fancybox=False,
                framealpha=0.9,
                edgecolor="0.8",
                borderpad=0.4,
                handlelength=2.2,
            )

        if has_data_for_tag:
            safe_tag_name = tag.replace("/", "_")
            png_path = os.path.join(save_dir, f"combined_{safe_tag_name}.png")
            pdf_path = os.path.join(save_dir, f"combined_{safe_tag_name}.pdf")
            plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
            plt.savefig(png_path, bbox_inches="tight")
            plt.savefig(pdf_path, bbox_inches="tight")
            print(f"✅ Successfully plotted and saved: {png_path}")
            print(f"✅ Successfully plotted and saved: {pdf_path}")
        else:
            print(f"⚠️ Tag '{tag}' not found in any task. Skipped.")
        plt.close(fig)


def plot_combined_tasks_mixed(
    task_exp_dicts,
    task_tag_map,
    save_dir,
    smooth_weight,
    layout="2x2",
    qmix_stride=1,
    include_list=None,
    max_step=None,
    task_max_step_map=None,
    report_events=None,
):
    """Plot N tasks into one combined figure where each task can use a different scalar tag."""
    os.makedirs(save_dir, exist_ok=True)
    rows, cols = parse_layout(layout)
    task_names = list(task_exp_dicts.keys())

    if len(task_names) > rows * cols:
        raise ValueError(
            f"Layout {layout} has only {rows * cols} panels, "
            f"but {len(task_names)} task_names were provided."
        )

    merged_exp_dict = merge_exp_dicts(task_exp_dicts.values())
    method_order, color_map = build_method_order_and_color_map(
        merged_exp_dict, include_list=include_list
    )

    print("\n🎨 Global fixed canonical color mapping:")
    for method_name in method_order:
        print(f"  - {method_name} -> {color_map[method_name]}")

    print("\n🧭 Per-task scalar tags:")
    for task_name in task_names:
        print(f"  - {task_name}: {task_tag_map[task_name]}")

    if task_max_step_map:
        print("\n⏱️ Per-task max steps:")
        for task_name in task_names:
            print(f"  - {task_name}: {task_max_step_map.get(task_name, max_step)}")

    fig_width = max(4.6 * cols, 6.5)
    fig_height = max(3.4 * rows, 4.2)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    axes = np.array(axes).reshape(-1)
    has_any_data = False

    for panel_idx, task_name in enumerate(task_names):
        ax = axes[panel_idx]
        row_idx = panel_idx // cols
        tag = task_tag_map[task_name]
        task_max_step = task_max_step_map.get(task_name, max_step) if task_max_step_map else max_step

        has_data = plot_tag_on_ax(
            ax=ax,
            exp_dict=task_exp_dicts[task_name],
            tag=tag,
            method_order=method_order,
            color_map=color_map,
            smooth_weight=smooth_weight,
            qmix_stride=qmix_stride,
            max_step=task_max_step,
            show_xlabel=(row_idx == rows - 1),
            show_ylabel=False,
            task_name=task_name,
            report_events=report_events,
        )

        has_any_data = has_any_data or has_data
        ax.set_title(infer_task_title(task_name), pad=8, fontweight="semibold")

        if not has_data:
            ax.text(
                0.5,
                0.5,
                f"No data for\n{tag}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
            )

    for empty_idx in range(len(task_names), rows * cols):
        axes[empty_idx].axis("off")

    handles, labels = collect_ordered_handles_labels(axes, method_order)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=min(len(labels), max(1, len(labels))),
            frameon=True,
            fancybox=False,
            framealpha=0.9,
            edgecolor="0.8",
            borderpad=0.4,
            handlelength=2.2,
        )

    if has_any_data:
        png_path = os.path.join(save_dir, "combined_mixed_metrics.png")
        pdf_path = os.path.join(save_dir, "combined_mixed_metrics.pdf")
        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
        plt.savefig(png_path, bbox_inches="tight")
        plt.savefig(pdf_path, bbox_inches="tight")
        print(f"✅ Successfully plotted and saved: {png_path}")
        print(f"✅ Successfully plotted and saved: {pdf_path}")
    else:
        print("⚠️ No requested task-specific tags were found. Skipped mixed-metric figure.")

    plt.close(fig)


# =========================
# Main
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot HARL results from TensorBoard logs with multi-seed shading."
    )

    parser.add_argument("--dirs", type=str, nargs="+", required=True)
    parser.add_argument("--include", type=str, nargs="+", default=None)
    parser.add_argument("--save_dir", type=str, default="./plots")
    parser.add_argument(
        "--smooth",
        type=float,
        default=0.90,
        help="EMA smoothing factor from 0.0 to 1.0. Default is 0.90.",
    )
    parser.add_argument("--qmix_stride", type=int, default=1)
    parser.add_argument("--max_step", type=float, default=None)
    parser.add_argument(
        "--task_names",
        type=str,
        nargs="+",
        default=None,
        help="Optional: combine multiple tasks into one figure. Task names are matched as substrings in --dirs.",
    )
    parser.add_argument("--layout", type=str, default="2x2")
    parser.add_argument(
        "--task_tags",
        type=str,
        nargs="+",
        default=None,
        help="Optional: scalar tag for each task in --task_names, in the same order.",
    )
    parser.add_argument(
        "--task_max_steps",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional: max_step for each task in --task_names, in the same order. "
            "Use none/null/-1 for no per-task override. These values override --max_step in combined-task mode."
        ),
    )

    args = parser.parse_args()

    target_tags = [
        "eval_average_episode_rewards",
        "train_episode_rewards",
        "eval_win_rate",
        "eval_score_rate",
    ]

    report_events = []

    print(f"🔍 Scanning {len(args.dirs)} directory paths...")
    if args.include:
        print(f"🎯 Filtering active. Only keeping experiments named: {args.include}")

    if args.task_names:
        print(f"🧩 Combined-task mode. Tasks: {args.task_names}")
        print(f"🧩 Layout: {args.layout}")

        if args.task_tags is not None and len(args.task_tags) != len(args.task_names):
            raise ValueError(
                f"--task_tags must have the same length as --task_names: "
                f"got {len(args.task_tags)} values for {len(args.task_names)} tasks."
            )

        parsed_task_max_steps = parse_optional_float_list(
            args.task_max_steps,
            expected_len=len(args.task_names),
            arg_name="--task_max_steps",
        )
        task_max_step_map = None
        if parsed_task_max_steps is not None:
            task_max_step_map = dict(zip(args.task_names, parsed_task_max_steps))

        task_to_dirs = group_dirs_by_task_name(root_dirs=args.dirs, task_names=args.task_names)
        task_exp_dicts = {}

        for task_name in args.task_names:
            task_dirs = task_to_dirs[task_name]
            if not task_dirs:
                print(f"⚠️ No --dirs path contains task name '{task_name}'. This task will be skipped.")
                continue

            exp_dict = get_experiment_groups(task_dirs, include_list=args.include)
            if not exp_dict:
                print(f"⚠️ No valid experiment logs found for task '{task_name}'. This task will be skipped.")
                continue

            task_exp_dicts[task_name] = exp_dict
            print(f"\n📊 Task '{task_name}' found {len(exp_dict)} experiments:")
            for exp_name, seeds in exp_dict.items():
                print(f"  - {exp_name}: {len(seeds)} total runs/seeds")

        if not task_exp_dicts:
            print("❌ No valid task logs found matching your criteria. Please check your paths, --task_names, or --include names.")
            sys.exit(1)

        print("\n🚀 Starting combined-task data extraction and plotting...")

        if args.task_tags is not None:
            task_tag_map = {
                task_name: tag
                for task_name, tag in zip(args.task_names, args.task_tags)
                if task_name in task_exp_dicts
            }
            plot_combined_tasks_mixed(
                task_exp_dicts=task_exp_dicts,
                task_tag_map=task_tag_map,
                save_dir=args.save_dir,
                smooth_weight=args.smooth,
                layout=args.layout,
                qmix_stride=args.qmix_stride,
                include_list=args.include,
                max_step=args.max_step,
                task_max_step_map=task_max_step_map,
                report_events=report_events,
            )
        else:
            plot_combined_tasks(
                task_exp_dicts=task_exp_dicts,
                target_tags=target_tags,
                save_dir=args.save_dir,
                smooth_weight=args.smooth,
                layout=args.layout,
                qmix_stride=args.qmix_stride,
                include_list=args.include,
                max_step=args.max_step,
                task_max_step_map=task_max_step_map,
                report_events=report_events,
            )

    else:
        exp_dict = get_experiment_groups(args.dirs, include_list=args.include)
        if not exp_dict:
            print("❌ No valid experiment logs found matching your criteria. Please check your paths or --include names.")
            sys.exit(1)

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
            report_events=report_events,
        )

    write_curve_report(args.save_dir, report_events)
    print("🎉 All done!")
