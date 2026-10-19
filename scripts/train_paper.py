#!/usr/bin/env python3
"""Launch the paper configurations; --dry-run needs only the standard library."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TASKS = {
    "grf_3v1": ("football", "academy_3_vs_1_with_keeper", 25_000_000),
    "grf_counterattack": ("football", "academy_counterattack_hard", 50_000_000),
    "humanoid": ("mamujoco", "Humanoid-v2-17x1", 10_000_000),
    "ant": ("mamujoco", "Ant-v2-4x2", 15_000_000),
    "walker": ("mamujoco", "Walker2d-v2-6x1", 10_000_000),
    "hand_over": ("dexhands", "ShadowHandOver", 50_000_000),
}
METHODS = {
    "prl": ("prl",),
    "happo": ("happo",),
    "happo_deep": ("happo_deep",),
    "mappo": ("mappo",),
    "mappo_prl": ("mappo_prl_noshare", "prl_mappo_noshare"),
    "mappo_deep": ("mappo_deep_noshare",),
    "current_embedding": ("pred_current_state",),
    "joint_action": ("pred_teammate_action",),
}


def config_path(task, method):
    env, name, _ = TASKS[task]
    for folder in METHODS[method]:
        path = ROOT / "tuned_configs" / env / name / folder / "config.json"
        if path.is_file():
            return path
    raise ValueError(f"No archived {method} configuration for {task}.")


def build_config(task, method, seed=1, output="results/paper", device=None):
    path = config_path(task, method)
    config = json.loads(path.read_text())
    algo = config["algo_args"]
    # Use the verified paper horizons, including 15M steps for Ant.
    # These agree with the released curated configurations.
    algo["train"]["num_env_steps"] = TASKS[task][2]
    algo["seed"].update(seed=seed, seed_specify=True)
    algo["logger"]["log_dir"] = str(Path(output).expanduser().resolve())
    config["main_args"].update(exp_name=method, load_config=str(path))
    if device is not None:
        algo["device"]["cuda"] = device == "cuda"
    if config["main_args"]["env"] == "dexhands":
        algo["eval"]["use_eval"] = False
        algo["train"]["episode_length"] = config["env_args"]["hands_episode_length"]
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List archived task/method pairs.")
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--method", choices=METHODS, default="prl")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", default="results/paper")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--dry-run", action="store_true", help="Print effective config without importing ML dependencies.")
    parser.add_argument("--num-env-steps", type=int, help="Override paper horizon (e.g. a smoke run).")
    parser.add_argument("--rollout-threads", type=int, help="Override training parallelism.")
    parser.add_argument("--eval-threads", type=int)
    parser.add_argument("--eval-episodes", type=int)
    parser.add_argument("--no-eval", action="store_true")
    opts = parser.parse_args()
    if opts.list:
        for task in TASKS:
            available = []
            for method in METHODS:
                try:
                    config_path(task, method)
                    available.append(method)
                except ValueError:
                    pass
            print(f"{task}: {', '.join(available)}")
        return
    if not opts.task:
        parser.error("--task is required unless --list is used")
    try:
        config = build_config(opts.task, opts.method, opts.seed, opts.output, opts.device)
    except ValueError as exc:
        parser.error(str(exc))
    algo = config["algo_args"]
    for option, section, key in (
        (opts.num_env_steps, "train", "num_env_steps"),
        (opts.rollout_threads, "train", "n_rollout_threads"),
        (opts.eval_threads, "eval", "n_eval_rollout_threads"),
        (opts.eval_episodes, "eval", "eval_episodes"),
    ):
        if option is not None:
            if option < 1:
                parser.error(f"{key} must be positive")
            algo[section][key] = option
    if opts.no_eval:
        algo["eval"]["use_eval"] = False
    batch_size = algo["train"]["episode_length"] * algo["train"]["n_rollout_threads"]
    if algo["train"]["num_env_steps"] < batch_size:
        parser.error(f"num_env_steps must allow at least one rollout ({batch_size} steps)")
    if opts.dry_run:
        print(json.dumps(config, indent=2))
        return
    if config["main_args"]["env"] == "dexhands":
        if opts.device == "cpu":
            parser.error("The archived Bi-DexHands configuration requires CUDA/Isaac Gym.")
        import isaacgym  # Must precede PyTorch.
    from harl.runners import RUNNER_REGISTRY
    runner = RUNNER_REGISTRY[config["main_args"]["algo"]](
        config["main_args"], algo, config["env_args"]
    )
    try:
        runner.run()
    finally:
        runner.close()


if __name__ == "__main__":
    main()
