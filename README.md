# MLDS Assignment 2: Deep Q-Network

Reproducible PyTorch implementation of DQN for `CartPole-v1` and `ALE/MsPacman-v5`.
It includes replay memory, an online/target network pair, Huber loss, epsilon scheduling,
checkpointing, assignment-required figures, 500-episode evaluation, and quantitative
validation against a random policy.

## Deepdish setup

Connect from the Northwestern network or VPN:

```bash
ssh vll0756@mlds-deepdish4.ads.northwestern.edu
cd ~/mlds-assignment2-dqn
bash scripts/setup_deepdish.sh
```

The setup script defaults to the CUDA 12.4 PyTorch index used on Deepdish 3/4. If an
administrator specifies a different supported wheel for another node, override it with
`PYTORCH_INDEX_URL=... bash scripts/setup_deepdish.sh`.

Before choosing a GPU, run `nvidia-smi`. Use only one card and select an idle card with
`CUDA_VISIBLE_DEVICES`; do not start MsPacman when every card is heavily occupied.

## Tests and static checks

```bash
.venv/bin/python -m pytest
.venv/bin/ruff check .
```

The tests specifically check terminal bootstrapping, replay wraparound and dtype,
assignment image preprocessing, target-network synchronization, network shapes,
metric persistence, plotting, checkpoint evaluation, and a short end-to-end CartPole run.

## Training

CartPole is inexpensive and intentionally runs on CPU:

```bash
bash scripts/run_cartpole.sh
```

MsPacman uses one selected GPU. Replace `2` with a currently idle GPU index:

```bash
GPU_ID=2 bash scripts/run_mspacman.sh
```

The launch script rechecks the selected GPU immediately before starting and refuses to
run when it sees at least 2 GiB allocated, at least 10% utilization, or any compute process.

Equivalent direct commands:

> These commands show the underlying CLI. On the shared server, prefer
> `GPU_ID=... bash scripts/run_mspacman.sh` because that wrapper performs the final
> idle-GPU preflight.

```bash
.venv/bin/python -m dqn_assignment train --env cartpole --episodes 600 \
  --output runs/cartpole_seed42 --device cpu

CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m dqn_assignment train --env mspacman \
  --episodes 6000 --output runs/mspacman_seed42 --device cuda
```

Important defaults match the assignment: CartPole uses `gamma=0.95`; MsPacman uses
`gamma=0.99`, the provided `88 x 80 x 1` preprocessing, and nine actions. Four recent
frames are stacked so the state encodes motion. MsPacman stores the stacks as `int8` in
replay and converts them to float only when sampled. At the default 50,000-transition
capacity, state/next-state storage is approximately 2.8 GB of host RAM.

Run long jobs inside `tmux` so the process survives a dropped SSH connection:

```bash
tmux new -s mspacman
GPU_ID=2 bash scripts/run_mspacman.sh
# Detach with Ctrl-b, then d. Reattach later with: tmux attach -t mspacman
```

If a run is interrupted, warm-resume the model, target network, optimizer, counters, and
metrics. Replay starts empty and is refilled before optimization resumes:

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m dqn_assignment train --env mspacman \
  --episodes 6000 --output runs/mspacman_seed42 --device cuda \
  --resume runs/mspacman_seed42/checkpoints/episode_02000.pt
```

## Required 500-episode evaluation

```bash
bash scripts/evaluate_all.sh
```

Each evaluation directory contains `summary.json`, all raw rewards, their mean and
population standard deviation, and `reward_histogram.png`.

## Required and non-standard plots

Training automatically generates:

- `episode_rewards.png`: raw reward plus the trailing 100-episode average.
- `max_q_values.png`: mean per-episode maximum Q-value.
- `optimization_diagnostics.png`: Huber loss and absolute TD error.
- `action_frequency_heatmap.png`: non-standard plot showing policy-action frequency.

Regenerate them without retraining:

```bash
.venv/bin/python -m dqn_assignment plot --run-dir runs/cartpole_seed42
```

## Quantitative validation

Run a trained-versus-random comparison and inspect the first ten greedy decisions:

```bash
.venv/bin/python -m dqn_assignment validate \
  --checkpoint runs/cartpole_seed42/checkpoints/final.pt \
  --episodes 30 --trace-steps 10 --device cpu

CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m dqn_assignment validate \
  --checkpoint runs/mspacman_seed42/checkpoints/final.pt \
  --episodes 30 --trace-steps 10 --device cuda
```

The command creates:

- `summary.json` with trained/random reward means, standard deviations, and raw rewards.
- `trained_vs_random.png` for quantitative policy comparison.
- `first_steps_trace.csv` with state statistics, action, maximum Q-value, reward, and
  termination flags for the first steps.

For an ablation, repeat training with only one parameter changed and a separate output
directory. For example, test target-network stability:

```bash
.venv/bin/python -m dqn_assignment train --env cartpole --episodes 300 \
  --target-update-interval 50 --output runs/ablation_target_50 --device cpu
.venv/bin/python -m dqn_assignment train --env cartpole --episodes 300 \
  --target-update-interval 2000 --output runs/ablation_target_2000 --device cpu
```

Other supported overrides include `--learning-rate`, `--batch-size`, `--warmup-steps`,
`--epsilon-decay-steps`, and `--replay-capacity`.

## Edge-case decisions to describe in the report

- `terminated=True` suppresses Bellman bootstrapping; a time-limit `truncated=True` does not.
- The agent never samples replay until at least one full batch exists.
- Target parameters are synchronized at initialization and at a fixed environment-step interval.
- Evaluation uses `epsilon=0` and seeds separate from training.
- CartPole's 500-step time limit is treated as successful truncation, not failure.
- MsPacman observations are shape-checked before preprocessing and stored as `int8`.
- Checkpoints and CSV metrics are written through temporary files before replacement.

See `report/REPORT_CHECKLIST.md` before submission.

