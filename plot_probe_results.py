import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TARGET_NAME_MAP = {
    "partial_observation_state": "Partial Obs. Uncertainty",
    "teammate_action": "Teammate Action Uncertainty",
    "transition_next_critic_latent": "Transition Uncertainty",
}

STAGE_ORDER = {
    "early": 0,
    "middle": 1,
    "final": 2,
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_and_summarize(csv_path, probe_type="linear"):
    df = pd.read_csv(csv_path)

    if probe_type is not None:
        df = df[df["probe_type"] == probe_type].copy()

    df["target_pretty"] = df["target_name"].map(TARGET_NAME_MAP).fillna(df["target_name"])

    summary = (
        df.groupby(["method", "stage", "target_name", "target_pretty", "probe_type"], as_index=False)
        .agg(
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            norm_mse_mean=("norm_mse", "mean"),
            norm_mse_std=("norm_mse", "std"),
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            n_agents=("agent_id", "nunique"),
        )
    )

    summary["stage_order"] = summary["stage"].map(STAGE_ORDER).fillna(999)
    summary = summary.sort_values(["target_name", "stage_order", "method"]).reset_index(drop=True)

    return df, summary


def plot_final_r2_bar(summary, out_dir):
    final_df = summary[summary["stage"] == "final"].copy()
    if len(final_df) == 0:
        print("[Warn] No final stage data found, skip final R2 bar plot.")
        return

    targets = list(final_df["target_pretty"].drop_duplicates())
    methods = list(final_df["method"].drop_duplicates())

    x = np.arange(len(targets))
    width = 0.8 / max(len(methods), 1)

    plt.figure(figsize=(10, 6))

    for i, method in enumerate(methods):
        sub = final_df[final_df["method"] == method]
        means = []
        stds = []
        for t in targets:
            row = sub[sub["target_pretty"] == t]
            if len(row) == 0:
                means.append(np.nan)
                stds.append(0.0)
            else:
                means.append(row["r2_mean"].values[0])
                stds.append(0.0 if pd.isna(row["r2_std"].values[0]) else row["r2_std"].values[0])

        plt.bar(x + i * width - (len(methods) - 1) * width / 2, means, width=width, yerr=stds, label=method)

    plt.xticks(x, targets, rotation=15)
    plt.ylabel("R^2 (higher is better)")
    plt.title("Final-stage Probe Performance (R^2)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "final_r2_bar.png"), dpi=200)
    plt.close()


def plot_final_norm_mse_bar(summary, out_dir):
    final_df = summary[summary["stage"] == "final"].copy()
    if len(final_df) == 0:
        print("[Warn] No final stage data found, skip final norm_mse bar plot.")
        return

    targets = list(final_df["target_pretty"].drop_duplicates())
    methods = list(final_df["method"].drop_duplicates())

    x = np.arange(len(targets))
    width = 0.8 / max(len(methods), 1)

    plt.figure(figsize=(10, 6))

    for i, method in enumerate(methods):
        sub = final_df[final_df["method"] == method]
        means = []
        stds = []
        for t in targets:
            row = sub[sub["target_pretty"] == t]
            if len(row) == 0:
                means.append(np.nan)
                stds.append(0.0)
            else:
                means.append(row["norm_mse_mean"].values[0])
                stds.append(0.0 if pd.isna(row["norm_mse_std"].values[0]) else row["norm_mse_std"].values[0])

        plt.bar(x + i * width - (len(methods) - 1) * width / 2, means, width=width, yerr=stds, label=method)

    plt.xticks(x, targets, rotation=15)
    plt.ylabel("Normalized MSE (lower is better)")
    plt.title("Final-stage Probe Performance (Normalized MSE)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "final_norm_mse_bar.png"), dpi=200)
    plt.close()


def plot_stage_curves(summary, out_dir):
    stage_df = summary.copy()
    unique_stages = set(stage_df["stage"].unique())

    if len(unique_stages.intersection({"early", "middle", "final"})) < 2:
        print("[Warn] Not enough stages for stage curve plot, skip.")
        return

    targets = list(stage_df["target_pretty"].drop_duplicates())
    methods = list(stage_df["method"].drop_duplicates())
    ordered_stages = [s for s in ["early", "middle", "final"] if s in unique_stages]

    for target in targets:
        sub_target = stage_df[stage_df["target_pretty"] == target].copy()

        # R2 curve
        plt.figure(figsize=(8, 5))
        for method in methods:
            sub = sub_target[sub_target["method"] == method]
            y = []
            yerr = []
            for stage in ordered_stages:
                row = sub[sub["stage"] == stage]
                if len(row) == 0:
                    y.append(np.nan)
                    yerr.append(0.0)
                else:
                    y.append(row["r2_mean"].values[0])
                    yerr.append(0.0 if pd.isna(row["r2_std"].values[0]) else row["r2_std"].values[0])

            plt.errorbar(ordered_stages, y, yerr=yerr, marker="o", label=method)

        plt.ylabel("R^2 (higher is better)")
        plt.title(f"Stage-wise R^2: {target}")
        plt.legend()
        plt.tight_layout()
        save_name = target.lower().replace(" ", "_").replace(".", "").replace("/", "_")
        plt.savefig(os.path.join(out_dir, f"stage_curve_r2_{save_name}.png"), dpi=200)
        plt.close()

        # norm_mse curve
        plt.figure(figsize=(8, 5))
        for method in methods:
            sub = sub_target[sub_target["method"] == method]
            y = []
            yerr = []
            for stage in ordered_stages:
                row = sub[sub["stage"] == stage]
                if len(row) == 0:
                    y.append(np.nan)
                    yerr.append(0.0)
                else:
                    y.append(row["norm_mse_mean"].values[0])
                    yerr.append(0.0 if pd.isna(row["norm_mse_std"].values[0]) else row["norm_mse_std"].values[0])

            plt.errorbar(ordered_stages, y, yerr=yerr, marker="o", label=method)

        plt.ylabel("Normalized MSE (lower is better)")
        plt.title(f"Stage-wise Normalized MSE: {target}")
        plt.legend()
        plt.tight_layout()
        save_name = target.lower().replace(" ", "_").replace(".", "").replace("/", "_")
        plt.savefig(os.path.join(out_dir, f"stage_curve_nmse_{save_name}.png"), dpi=200)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to probe_results_all.csv")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for plots")
    parser.add_argument("--probe_type", type=str, default="linear", help="Probe type to visualize")
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    raw_df, summary = load_and_summarize(args.csv, probe_type=args.probe_type)

    raw_df.to_csv(os.path.join(args.out_dir, "raw_filtered.csv"), index=False)
    summary.to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)

    plot_final_r2_bar(summary, args.out_dir)
    plot_final_norm_mse_bar(summary, args.out_dir)
    plot_stage_curves(summary, args.out_dir)

    print(f"Done. Results saved to: {args.out_dir}")
    print(f"- raw_filtered.csv")
    print(f"- summary.csv")
    print(f"- final_r2_bar.png")
    print(f"- final_norm_mse_bar.png")
    print(f"- stage_curve_*.png (if multiple stages exist)")


if __name__ == "__main__":
    main()