# Reproducing the PRL experiments

Run commands from the repository root. Training uses the original HARL interfaces;
`scripts/train_paper.py` selects the archived configuration, sets the seed/output
path, and applies the common analysis horizon. It prints the exact effective JSON
with `--dry-run` before any ML dependency is imported.

## Install

Use Linux for benchmark training. Keep separate environments for GRF, MA-MuJoCo,
and Bi-DexHands because the simulators require different native dependencies.
Python 3.8 and PyTorch 2.0.1 are suitable for the legacy Isaac Gym stack. Python
3.10 was also used for the GRF release smoke check. These are verification
stacks, not a claim that every archived experiment used identical package versions.

```bash
conda create -n prl python=3.8
conda activate prl
# Install the PyTorch build appropriate for the driver/CUDA version first.
python -m pip install 'torch==2.0.1'
python -m pip install -r requirements/core.txt
python -m pip install -e .
python -m unittest discover -s tests -v
```

The core tests need no benchmark simulator or GPU. The installation is intended
for editable source use: the bundled simulator assets are read from the checkout.

Install **only the chosen benchmark** next:

- **GRF:** install the native libraries and game engine following
  [Google Research Football](https://github.com/google-research/football#installation),
  then `python -m pip install -r requirements/football.txt`.
- **MA-MuJoCo:** install MuJoCo **2.1.0** and the native libraries described in
  [mujoco-py](https://github.com/openai/mujoco-py#install-mujoco), then
  `python -m pip install -r requirements/mamujoco.txt`. The paper uses legacy
  `Humanoid-v2`, `Ant-v2`, and `Walker2d-v2`; replacing them with current Gymnasium
  tasks changes the benchmark.
- **Bi-DexHands:** install [NVIDIA Isaac Gym Preview 4](https://developer.nvidia.com/isaac-gym)
  and follow the [Bi-DexHands installation instructions](https://github.com/PKU-MARL/DexterousHands).
  Install the Isaac Gym Python package into the active environment, then
  `python -m pip install -r requirements/dexhands.txt`. The launcher imports
  Isaac Gym before PyTorch. The archived task requires CUDA and runs 256
  parallel environments; it is not a CPU benchmark.

Gym 0.21 may require an older pip/wheel toolchain; the upstream HARL setup also
supports `conda install -c conda-forge gym=0.21.0` when pip cannot build it.
The benchmark engines, NVIDIA libraries, and pretrained checkpoints are not
installed by `pip install -e .`.

## Main comparison

```bash
python scripts/train_paper.py --list
python scripts/train_paper.py --task grf_3v1 --method prl --seed 1 --dry-run
python scripts/train_paper.py --task grf_3v1 --method prl --seed 1
```

| Task alias | Benchmark task | Analysis horizon | PRL coefficient |
|---|---|---:|---:|
| `grf_3v1` | academy_3_vs_1_with_keeper | 25M | 0.5 |
| `grf_counterattack` | academy_counterattack_hard | 50M | 1.0 |
| `humanoid` | Humanoid-v2, 17x1 | 10M | 1.0 |
| `ant` | Ant-v2, 4x2 | 15M | 1.0 |
| `walker` | Walker2d-v2, 6x1 | 10M | 1.0 |
| `hand_over` | ShadowHandOver | 50M | 1.0 |

The four main methods are `prl`, `happo`, `happo_deep`, and `mappo`. The Ant
comparison uses 15M steps. The launcher preserves the curated configurations
and original network/loss hyperparameters.

For the ten-seed main comparison, run the following separately in each
benchmark's environment, choosing a task from the table:

```bash
for method in prl happo happo_deep mappo; do
  for seed in 1 2 3 4 5 6 7 8 9 10; do
    python scripts/train_paper.py --task grf_3v1 --method "$method" --seed "$seed"
  done
done
```

The loop is sequential. To schedule independent runs across GPUs, set
`CUDA_VISIBLE_DEVICES` for each process and use distinct seeds. Training outputs
are placed under `results/paper/<environment>/<task>/<algorithm>/<method>/seed-*`.
Each run saves its effective `config.json`, TensorBoard logs, and model files.
Use `--output` to select another root and `tensorboard --logdir results/paper`
to inspect logs. The standard JSON config and `examples/train.py --load_config`
interface remain available for additional experiments.

## Architecture and auxiliary controls

PRL and the deep baselines use width 256 and four residual blocks; each residual
block has four Linear–LayerNorm–SiLU units. The policy head consumes the actor
encoder feature. Published PRL configurations have `condition_on_pred=false`:
the predicted embedding is an auxiliary training output, not an extra policy
input. The critic produces a next-state representation under `torch.no_grad()`;
actor updates also detach the supplied target. There is no target-network/EMA
teacher in these experiments.

The implementation takes the mean squared error over embedding dimensions and
then averages over active samples. HAPPO defaults to masked auxiliary loss and
target layer normalization. Humanoid explicitly disables target normalization;
its PRL results therefore use the unnormalized critic embedding. The MAPPO-PRL
configs set the corresponding masking and normalization flags explicitly.

The following additional methods have archived configs for GRF 3v1, GRF hard,
Humanoid, and Hand Over (not Ant/Walker):

```bash
python scripts/train_paper.py --task humanoid --method mappo_prl --seed 1
python scripts/train_paper.py --task humanoid --method mappo_deep --seed 1
python scripts/train_paper.py --task humanoid --method current_embedding --seed 1
python scripts/train_paper.py --task humanoid --method joint_action --seed 1
```

`mappo_prl` and `mappo_deep` use unshared actor parameters. `current_embedding`
is the historical `pred_current_state` folder: it targets the **current critic
embedding**, not raw current state. `joint_action` is the historical
`pred_teammate_action` folder: the implementation concatenates **all agents'
actions, including the learner's own action**, and predicts them with MSE. These
names clarify the implemented controls rather than changing their targets.
All paper recipes are feed-forward. Extra recurrent configs/source branches are
retained research utilities and are not covered by the paper release tests.

## Training metrics

GRF uses evaluation win rate; MA-MuJoCo uses evaluation episode return.
Bi-DexHands uses the training reward statistic emitted by the Isaac Gym logger,
not a separate evaluation rollout. Do not relabel its curve as evaluation return.
Use the saved TensorBoard logs from your own training runs to inspect learning
curves. Final-10 averages the last ten logged checkpoints for each run; compare
methods using the same task budgets and matched training seeds.

## Release verification

```bash
python -m unittest discover -s tests -v
# A genuine environment rollout/update; 200 steps is a smoke test, not a result.
python scripts/train_paper.py --task grf_3v1 --method prl --seed 1 \
  --device cpu --rollout-threads 1 --num-env-steps 200 --no-eval
```

The unit tests check target-gradient isolation from the critic, masked normalized
MSE, encoder/predictor updates, discrete and continuous action shapes, disabled
and joint-action auxiliary losses, target time alignment, and all 40 archived
recipe combinations. Core CPU checks used Python 3.8, imported PyTorch 2.0.1,
and NumPy 1.23.5. The real GRF smoke runs used a pre-existing Python 3.10 / PyTorch
2.11.0 / NumPy 2.2.6 / GRF 2.10.2 environment; that separate check does not replace
the conservative legacy dependency recipes above. Full 10-seed training and
end-to-end MA-MuJoCo/Bi-DexHands training were not rerun during release preparation.

## Source provenance and release-only changes

The released implementation is the author's curated PRL source snapshot, rather
than the older public snapshot whose MAPPO auxiliary interface was incomplete.
This preserves the research loss definitions, buffers, actor models, and runners.
Release-only edits replace `eval` with `ast.literal_eval` for CLI values, correct
JSON-serialization exception handling, remove an unreachable malformed actor
initialization branch, add explicit dependency declarations and the paper launcher,
and remove machine-specific result symlinks and unrelated plotting outputs.
The launcher changes no network/loss hyperparameters and uses the verified
paper horizons, including 15M steps for Ant. Optional experimental algorithms
in the source are retained for compatibility and are not reported as paper baselines.
