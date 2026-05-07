from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
import matplotlib.pyplot as plt


TARGET_ORDER = [
    "partial_observation_state",
    "teammate_action",
    "transition_next_critic_latent",
]

TARGET_DISPLAY = {
    "partial_observation_state": "State Reconstruction",
    "teammate_action": "Teammate-Action Reconstruction",
    "transition_next_critic_latent": "Next Critic Latent Reconstruction",
}

METHOD_ORDER = ["HAPPO", "MAPPO", "PRL"]

METHOD_COLOR = {
    "HAPPO": "#D7E3F3",
    "MAPPO": "#6F93C5",
    "PRL": "#2F5597",
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=str,
        default="probe_results_humanoid_ckpt",
        help="Root directory produced by run_humanoid_probe_ckpt_sweep.sh.",
    )

    parser.add_argument(
        "--metric",
        type=str,
        default="r2",
        choices=["r2", "norm_mse"],
    )

    parser.add_argument(
        "--probe_type",
        type=str,
        default="linear",
        choices=["linear", "mlp"],
    )

    parser.add_argument(
        "--save_name",
        type=str,
        default="humanoid_probe_training_dynamics",
    )

    return parser.parse_args()


def parse_dir_name(name: str):
    """
    Expected:
      mamujoco_humanoid_17x1_seed1_ep500
      mamujoco_humanoid_17x1_seed2_ep1000
    """
    m = re.match(r"mamujoco_humanoid_17x1_seed(\d+)_ep(\d+)$", name)
    if m is None:
        return None

    seed = int(m.group(1))
    checkpoint = int(m.group(2))
    return seed, checkpoint


def load_results(root: Path, probe_type: str) -> pd.DataFrame:
    rows = []

    for run_dir in sorted(root.glob("mamujoco_humanoid_17x1_seed*_ep*")):
        if not run_dir.is_dir():
            continue

        parsed = parse_dir_name(run_dir.name)
        if parsed is None:
            continue

        seed, checkpoint = parsed

        csvs = sorted(run_dir.glob("probe_results_*.csv"))
        csvs = [
            p for p in csvs
            if p.name != "probe_results_all.csv" and not p.name.endswith("_summary.csv")
        ]

        if len(csvs) == 0:
            all_csv = run_dir / "probe_results_all.csv"
            if all_csv.exists():
                csvs = [all_csv]

        for csv_path in csvs:
            df = pd.read_csv(csv_path)

            if "probe_type" in df.columns:
                df = df[df["probe_type"] == probe_type].copy()

            if len(df) == 0:
                continue

            df["seed"] = seed
            df["checkpoint"] = checkpoint
            df["source_csv"] = str(csv_path)

            rows.append(df)

    if len(rows) == 0:
        raise RuntimeError(f"No valid probe CSV files found under: {root}")

    out = pd.concat(rows, ignore_index=True)

    print("\n[Info] Loaded rows:")
    print(
        out.groupby(["checkpoint", "seed", "method", "target_name"])
        .size()
        .reset_index(name="n_rows")
        .sort_values(["checkpoint", "seed", "method", "target_name"])
        .to_string(index=False)
    )

    return out


def aggregate(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    required = {"seed", "checkpoint", "method", "target_name", metric}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Step 1: average over agents within each seed/checkpoint/method/target.
    by_seed = (
        df.groupby(["checkpoint", "seed", "method", "target_name"], as_index=False)
        .agg(metric_value=(metric, "mean"))
    )

    # Step 2: aggregate over seeds.
    agg = (
        by_seed.groupby(["checkpoint", "method", "target_name"], as_index=False)
        .agg(
            mean=("metric_value", "mean"),
            std=("metric_value", "std"),
            n_seeds=("metric_value", "count"),
        )
    )

    agg["std"] = agg["std"].fillna(0.0)
    agg["target_display"] = agg["target_name"].map(TARGET_DISPLAY).fillna(agg["target_name"])

    print("\n[Info] Aggregated results:")
    print(
        agg.sort_values(["target_name", "checkpoint", "method"])
        .to_string(index=False)
    )

    return agg


def plot_grouped_bars(agg: pd.DataFrame, metric: str, save_pdf: Path, save_png: Path):
    checkpoints = sorted(agg["checkpoint"].unique())
    methods = [m for m in METHOD_ORDER if m in set(agg["method"])]

    plt.rcParams.update({
        "font.size": 11.5,
        "axes.titlesize": 12.5,
        "axes.labelsize": 12,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15.5, 4.2),
        sharey=True,
    )

    x = list(range(len(checkpoints)))
    width = 0.78 / max(len(methods), 1)

    for ax, target in zip(axes, TARGET_ORDER):
        sub = agg[agg["target_name"] == target].copy()

        for i, method in enumerate(methods):
            vals = []
            errs = []

            for ckpt in checkpoints:
                row = sub[
                    (sub["checkpoint"] == ckpt)
                    & (sub["method"] == method)
                ]

                if len(row) == 0:
                    vals.append(0.0)
                    errs.append(0.0)
                else:
                    vals.append(float(row["mean"].iloc[0]))
                    errs.append(float(row["std"].iloc[0]))

            xpos = [
                v + (i - (len(methods) - 1) / 2) * width
                for v in x
            ]

            ax.bar(
                xpos,
                vals,
                width=width,
                yerr=errs,
                label=method,
                color=METHOD_COLOR.get(method, "#999999"),
                edgecolor="black",
                linewidth=0.45,
                capsize=2.0,
                error_kw={
                    "elinewidth": 0.8,
                    "capthick": 0.8,
                },
            )

        ax.set_title(TARGET_DISPLAY.get(target, target), fontweight="semibold")
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in checkpoints], rotation=0)
        ax.set_xlabel("Checkpoint")
        ax.grid(axis="y", linestyle="--", alpha=0.25, linewidth=0.7)
        ax.set_axisbelow(True)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ylabel = "Probe R$^2$ (higher is better)" if metric == "r2" else "Normalized MSE (lower is better)"
    axes[0].set_ylabel(ylabel)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(methods),
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        columnspacing=1.2,
        handlelength=1.2,
        handletextpad=0.35,
    )

    plt.tight_layout(rect=[0.02, 0.02, 1.0, 0.94], w_pad=1.0)

    save_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_pdf, dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.savefig(save_png, dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close()

    print(f"\n[Saved] PDF: {save_pdf}")
    print(f"[Saved] PNG: {save_png}")


def main():
    args = parse_args()

    root = Path(args.root)
    out_dir = root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(root=root, probe_type=args.probe_type)
    agg = aggregate(df=df, metric=args.metric)

    save_csv = out_dir / f"{args.save_name}_{args.probe_type}_{args.metric}_aggregated.csv"
    agg.to_csv(save_csv, index=False)
    print(f"\n[Saved] Aggregated CSV: {save_csv}")

    save_pdf = out_dir / f"{args.save_name}_{args.probe_type}_{args.metric}.pdf"
    save_png = out_dir / f"{args.save_name}_{args.probe_type}_{args.metric}.png"

    plot_grouped_bars(
        agg=agg,
        metric=args.metric,
        save_pdf=save_pdf,
        save_png=save_png,
    )


if __name__ == "__main__":
    main()