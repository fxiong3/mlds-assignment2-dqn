"""Quantitative validation beyond unit tests, as required by the assignment."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dqn_assignment.agent import DQNAgent
from dqn_assignment.config import ExperimentConfig
from dqn_assignment.envs import ObservationStacker, adapt_observation, make_environment
from dqn_assignment.training import _build_network, _load_checkpoint, resolve_device


def _rollout(
    environment: Any,
    config: ExperimentConfig,
    *,
    episodes: int,
    seed_offset: int,
    agent: DQNAgent | None,
) -> list[float]:
    rewards: list[float] = []
    for episode_index in range(episodes):
        raw_observation, _ = environment.reset(seed=config.seed + seed_offset + episode_index)
        stacker = ObservationStacker(config.frame_stack)
        observation = stacker.reset(adapt_observation(config.environment_id, raw_observation))
        total_reward = 0.0
        max_steps = config.max_steps_per_episode or 1_000_000_000
        for _ in range(max_steps):
            if agent is None:
                action = int(environment.action_space.sample())
            else:
                action, _ = agent.select_action(observation, epsilon=0.0)
            next_raw, reward, terminated, truncated, _ = environment.step(action)
            observation = stacker.append(adapt_observation(config.environment_id, next_raw))
            total_reward += float(reward)
            if terminated or truncated:
                break
        rewards.append(total_reward)
    return rewards


def run_validation(
    checkpoint_path: Path,
    *,
    episodes: int = 30,
    trace_steps: int = 10,
    device: str = "auto",
    output_dir: Path | None = None,
) -> Path:
    """Compare a trained policy with random actions and trace early decisions."""

    if episodes <= 0 or trace_steps <= 0:
        raise ValueError("episodes and trace_steps must be positive")
    selected_device = resolve_device(device)
    checkpoint = _load_checkpoint(checkpoint_path, selected_device)
    config: ExperimentConfig = checkpoint["config"]
    destination = output_dir or checkpoint_path.parent.parent / "validation"
    destination.mkdir(parents=True, exist_ok=True)
    environment = make_environment(config)

    try:
        action_count = int(checkpoint["action_count"])
        agent = DQNAgent(
            _build_network(tuple(checkpoint["observation_shape"]), action_count),
            None,
            action_count,
            config.dqn,
            selected_device,
            seed=config.seed,
        )
        agent.load_state_dict(checkpoint["agent"])
        environment.action_space.seed(config.seed + 200_000)
        trained_rewards = _rollout(
            environment,
            config,
            episodes=episodes,
            seed_offset=200_000,
            agent=agent,
        )
        random_rewards = _rollout(
            environment,
            config,
            episodes=episodes,
            seed_offset=200_000,
            agent=None,
        )

        raw_observation, _ = environment.reset(seed=config.seed + 400_000)
        stacker = ObservationStacker(config.frame_stack)
        observation = stacker.reset(adapt_observation(config.environment_id, raw_observation))
        trace_path = destination / "first_steps_trace.csv"
        with trace_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "step",
                    "state_mean",
                    "state_std",
                    "state_min",
                    "state_max",
                    "action",
                    "max_q",
                    "reward",
                    "terminated",
                    "truncated",
                ],
            )
            writer.writeheader()
            for step in range(trace_steps):
                action, max_q = agent.select_action(observation, epsilon=0.0)
                next_raw, reward, terminated, truncated, _ = environment.step(action)
                writer.writerow(
                    {
                        "step": step,
                        "state_mean": float(np.mean(observation)),
                        "state_std": float(np.std(observation)),
                        "state_min": float(np.min(observation)),
                        "state_max": float(np.max(observation)),
                        "action": action,
                        "max_q": max_q,
                        "reward": float(reward),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                    }
                )
                observation = stacker.append(adapt_observation(config.environment_id, next_raw))
                if terminated or truncated:
                    break
    finally:
        environment.close()

    trained = np.asarray(trained_rewards, dtype=np.float64)
    random_values = np.asarray(random_rewards, dtype=np.float64)
    payload = {
        "environment": config.environment_id,
        "episodes_per_policy": episodes,
        "trained": {
            "mean_reward": float(trained.mean()),
            "standard_deviation": float(trained.std(ddof=0)),
            "rewards": trained.tolist(),
        },
        "random": {
            "mean_reward": float(random_values.mean()),
            "standard_deviation": float(random_values.std(ddof=0)),
            "rewards": random_values.tolist(),
        },
        "mean_reward_improvement": float(trained.mean() - random_values.mean()),
        "trace_file": "first_steps_trace.csv",
    }
    summary_path = destination / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    plt.figure(figsize=(7, 5))
    plt.boxplot([random_values, trained], tick_labels=["Random", "Trained"], showmeans=True)
    plt.ylabel("Episode reward")
    plt.title(f"{config.environment_id}: Trained Policy vs Random Baseline")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(destination / "trained_vs_random.png", dpi=180, bbox_inches="tight")
    plt.close()
    return summary_path
