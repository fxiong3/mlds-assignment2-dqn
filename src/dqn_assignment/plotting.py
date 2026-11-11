"""Assignment-required and validation plots."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dqn_assignment.metrics import EpisodeRecord, moving_average


def _finish_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_training(records: Sequence[EpisodeRecord], output_dir: Path) -> list[Path]:
    """Generate required training curves plus a non-standard action heatmap."""

    if not records:
        raise ValueError("At least one episode record is required for plotting")
    plot_dir = output_dir / "plots"
    episodes = np.asarray([record.episode for record in records])
    rewards = np.asarray([record.reward for record in records])
    mean_max_q = np.asarray([record.mean_max_q for record in records])

    reward_path = plot_dir / "episode_rewards.png"
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards, alpha=0.35, linewidth=0.8, label="Episode reward")
    plt.plot(
        episodes,
        moving_average(rewards, 100),
        linewidth=2.0,
        label="100-episode moving average",
    )
    plt.xlabel("Training episode")
    plt.ylabel("Episode reward")
    plt.title("Episode Rewards During DQN Training")
    plt.grid(alpha=0.25)
    plt.legend()
    _finish_figure(reward_path)

    q_path = plot_dir / "max_q_values.png"
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, mean_max_q, linewidth=1.0)
    plt.xlabel("Training episode")
    plt.ylabel("Mean of per-state maximum Q-value")
    plt.title("Maximum Q-Values During DQN Training")
    plt.grid(alpha=0.25)
    _finish_figure(q_path)

    diagnostics_path = plot_dir / "optimization_diagnostics.png"
    losses = np.asarray([record.mean_loss for record in records])
    td_errors = np.asarray([record.mean_absolute_td_error for record in records])
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(episodes, losses, linewidth=0.9)
    axes[0].set_ylabel("Mean Huber loss")
    axes[0].grid(alpha=0.25)
    axes[1].plot(episodes, td_errors, linewidth=0.9, color="tab:orange")
    axes[1].set_xlabel("Training episode")
    axes[1].set_ylabel("Mean absolute TD error")
    axes[1].grid(alpha=0.25)
    figure.suptitle("Optimization Diagnostics")
    _finish_figure(diagnostics_path)

    action_path = plot_dir / "action_frequency_heatmap.png"
    action_counts = np.asarray([record.action_counts for record in records], dtype=np.float64)
    totals = action_counts.sum(axis=1, keepdims=True)
    frequencies = np.divide(
        action_counts,
        totals,
        out=np.zeros_like(action_counts),
        where=totals > 0,
    )
    smoothed = np.column_stack(
        [moving_average(frequencies[:, action], 100) for action in range(frequencies.shape[1])]
    )
    x_min = float(episodes[0])
    x_max = float(episodes[-1])
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5
    plt.figure(figsize=(10, max(4, min(8, frequencies.shape[1] * 0.7))))
    image = plt.imshow(
        smoothed.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[x_min, x_max, -0.5, frequencies.shape[1] - 0.5],
        vmin=0.0,
        vmax=max(0.01, float(smoothed.max())),
        cmap="viridis",
    )
    plt.colorbar(image, label="100-episode moving action frequency")
    plt.yticks(
        range(frequencies.shape[1]),
        [f"Action {index}" for index in range(frequencies.shape[1])],
    )
    plt.xlabel("Training episode")
    plt.ylabel("Action")
    plt.title("Policy Action Frequency Over Training (Non-Standard Plot)")
    _finish_figure(action_path)

    return [reward_path, q_path, diagnostics_path, action_path]


def plot_evaluation_histogram(
    rewards: Sequence[float],
    output_path: Path,
    *,
    title: str,
) -> Path:
    """Plot the required rollout reward histogram with mean and standard deviation."""

    if not rewards:
        raise ValueError("At least one evaluation reward is required")
    values = np.asarray(rewards, dtype=np.float64)
    bins = min(30, max(5, int(np.sqrt(values.size))))
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins, color="tab:blue", alpha=0.8, edgecolor="white")
    plt.axvline(values.mean(), color="tab:red", linestyle="--", linewidth=2, label="Mean")
    plt.xlabel("Episode reward")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend(title=f"mean={values.mean():.2f}\nstd={values.std(ddof=0):.2f}")
    plt.grid(axis="y", alpha=0.2)
    _finish_figure(output_path)
    return output_path


def write_evaluation_summary(path: Path, rewards: Sequence[float]) -> Path:
    """Write inspectable raw rewards and aggregate evaluation statistics."""

    values = np.asarray(rewards, dtype=np.float64)
    payload = {
        "episodes": int(values.size),
        "mean_reward": float(values.mean()),
        "standard_deviation": float(values.std(ddof=0)),
        "minimum_reward": float(values.min()),
        "maximum_reward": float(values.max()),
        "rewards": values.tolist(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
