# make_main_table.py

import os
import re
import argparse
from collections import defaultdict, OrderedDict

import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def format_method_name(exp_name):
    """
    Clean method names for paper tables.

    Rules:
        PRL_ours / HAPPO_PRL / xxx_PRL_xxx -> PRL
        HAPPO_baseline -> HAPPO
        MAPPO_baseline -> MAPPO
    """
    if "PRL" in exp_name.upper():
        return "PRL"

    name = exp_name
    if name.endswith("_baseline"):
        name = name[:-len("_baseline")]

    return name.replace("_", " ")


def infer_task_name(root_dir):
    """
    Infer task name from root directory.

    Examples:
        results/grf_counterattack_hard_seed_1 -> grf_counterattack_hard
        results/smac_3s5z_seed_2 -> smac_3s5z
        results/mamujoco_ant_4x2_seed_1 -> mamujoco_ant_4x2
    """
    base = os.path.basename(os.path.normpath(root_dir))
    base = re.sub(r"_seed_\d+$", "", base)
    base = re.sub(r"-seed-\d+$", "", base)
    return base


def get_experiment_groups(root_dirs, include_list=None):
    """
    Return:
        task_dict[task_name][exp_name][seed_id] = [log_folders]
    """
    task_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    pattern = re.compile(
        r"(seed-\d+)-(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})"
    )

    for root_dir in root_dirs:
        task_name = infer_task_name(root_dir)

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

            seed_id = f"{root_dir}_{seed_str}"

            task_dict[task_name][exp_name][seed_id].append(
                (timestamp_str, tb_folder)
            )

    for task_name in task_dict:
        for exp_name in task_dict[task_name]:
            for seed_id in task_dict[task_name][exp_name]:
                task_dict[task_name][exp_name][seed_id].sort(key=lambda x: x[0])
                task_dict[task_name][exp_name][seed_id] = [
                    x[1] for x in task_dict[task_name][exp_name][seed_id]
                ]

    return task_dict


def read_tb_scalar(folder, tag):
    """
    Read one scalar tag from TensorBoard logs.
    Return:
        dict: step -> value
    """
    values = {}

    for root_path, dirs, files in os.walk(folder):
        if not any(f.startswith("events.out.tfevents") for f in files):
            continue

        try:
            ea = EventAccumulator(root_path, size_guidance={"scalars": 0})
            ea.Reload()

            available_scalars = ea.Tags().get("scalars", [])

            if tag not in available_scalars:
                continue

            for event in ea.Scalars(tag):
                values[event.step] = event.value

        except Exception as e:
            print(f"[Warning] Failed to read {root_path}: {e}")

    return values


def summarize_curve(step_to_value, last_k=10, mode="last"):
    """
    Summarize one seed curve into one scalar.

    mode:
        last      -> average over last_k points
        final     -> final point
        best      -> best point
        best_last -> best among last_k points
    """
    if not step_to_value:
        return np.nan

    items = sorted(step_to_value.items())
    values = np.array([v for _, v in items], dtype=float)

    if len(values) == 0:
        return np.nan

    if mode == "final":
        return values[-1]

    if mode == "best":
        return np.nanmax(values)

    if mode == "best_last":
        return np.nanmax(values[-last_k:])

    if mode == "last":
        return np.nanmean(values[-last_k:])

    raise ValueError(f"Unknown mode: {mode}")


def collect_results(task_dict, include_list, tag, last_k, mode, scale):
    """
    Return:
        raw_results[task][exp] = list of seed scalar values
    """
    raw_results = defaultdict(lambda: defaultdict(list))

    for task_name, exp_dict in task_dict.items():
        exp_order = include_list if include_list is not None else sorted(exp_dict.keys())

        for exp_name in exp_order:
            if exp_name not in exp_dict:
                continue

            for seed_id, folders in exp_dict[exp_name].items():
                merged = {}

                for folder in folders:
                    step_to_value = read_tb_scalar(folder, tag)
                    merged.update(step_to_value)

                scalar = summarize_curve(
                    merged,
                    last_k=last_k,
                    mode=mode,
                )

                if not np.isnan(scalar):
                    raw_results[task_name][exp_name].append(scalar * scale)

    return raw_results


def format_mean_std(values, precision=2):
    if len(values) == 0:
        return "--"

    values = np.array(values, dtype=float)
    mean = np.nanmean(values)

    if len(values) > 1:
        std = np.nanstd(values, ddof=0)
    else:
        std = 0.0

    return f"{mean:.{precision}f} ± {std:.{precision}f}"


def build_table(raw_results, include_list, precision=2, bold_best=True):
    """
    Build paper-style table.

    Rows: task
    Columns: methods
    Cells: mean ± std
    """
    rows = OrderedDict()

    all_tasks = list(raw_results.keys())

    for task_name in all_tasks:
        row = OrderedDict()

        exp_order = include_list
        if exp_order is None:
            exp_order = sorted(raw_results[task_name].keys())

        means = {}

        for exp_name in exp_order:
            values = raw_results[task_name].get(exp_name, [])
            if len(values) > 0:
                means[exp_name] = np.nanmean(values)

        best_exp = None
        if means:
            best_exp = max(means, key=means.get)

        for exp_name in exp_order:
            col_name = format_method_name(exp_name)
            values = raw_results[task_name].get(exp_name, [])

            cell = format_mean_std(values, precision=precision)

            if bold_best and exp_name == best_exp and cell != "--":
                cell = f"\\textbf{{{cell}}}"

            row[col_name] = cell

        rows[task_name] = row

    return pd.DataFrame.from_dict(rows, orient="index")


def save_table(df, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, f"{filename}.csv")
    tex_path = os.path.join(output_dir, f"{filename}.tex")

    df.to_csv(csv_path)

    latex = df.to_latex(
        escape=False,
        column_format="l" + "c" * len(df.columns),
    )

    with open(tex_path, "w") as f:
        f.write(latex)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved LaTeX: {tex_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate main result table from TensorBoard logs."
    )

    parser.add_argument(
        "--dirs",
        type=str,
        nargs="+",
        required=True,
        help=(
            "Root result directories. Usually one directory per seed, "
            "e.g. results/grf_counterattack_hard_seed_1 results/grf_counterattack_hard_seed_2"
        ),
    )

    parser.add_argument(
        "--include",
        type=str,
        nargs="+",
        required=True,
        help=(
            "Methods to include. Table column order follows this order, "
            "e.g. HAPPO_baseline HAPPO_deep_baseline PRL_ours MAPPO_baseline QMIX_baseline"
        ),
    )

    parser.add_argument(
        "--tag",
        type=str,
        default="eval_average_episode_rewards",
        help=(
            "TensorBoard scalar tag used for the table. "
            "Examples: eval_average_episode_rewards, eval_win_rate, eval_score_rate"
        ),
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="last",
        choices=["last", "final", "best", "best_last"],
        help=(
            "How to summarize each seed curve. "
            "last = average over last K points; final = final point; "
            "best = best over all points; best_last = best over last K points."
        ),
    )

    parser.add_argument(
        "--last_k",
        type=int,
        default=10,
        help="Number of final evaluation points used when mode is last or best_last.",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help=(
            "Scale values before table generation. "
            "Use 100 if your win rate / score rate is logged as 0-1 but you want percentage."
        ),
    )

    parser.add_argument(
        "--precision",
        type=int,
        default=2,
        help="Decimal precision in mean ± std.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./tables",
        help="Directory to save CSV and LaTeX tables.",
    )

    parser.add_argument(
        "--filename",
        type=str,
        default="main_results",
        help="Output filename without extension.",
    )

    parser.add_argument(
        "--no_bold_best",
        action="store_true",
        help="Disable bolding best result in each row.",
    )

    args = parser.parse_args()

    task_dict = get_experiment_groups(
        root_dirs=args.dirs,
        include_list=args.include,
    )

    if not task_dict:
        print("No valid TensorBoard logs found.")
        exit(1)

    raw_results = collect_results(
        task_dict=task_dict,
        include_list=args.include,
        tag=args.tag,
        last_k=args.last_k,
        mode=args.mode,
        scale=args.scale,
    )

    df = build_table(
        raw_results=raw_results,
        include_list=args.include,
        precision=args.precision,
        bold_best=not args.no_bold_best,
    )

    print("\nMain result table:")
    print(df)

    save_table(
        df=df,
        output_dir=args.output_dir,
        filename=args.filename,
    )