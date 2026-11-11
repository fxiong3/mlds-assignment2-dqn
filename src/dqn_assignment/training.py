"""End-to-end DQN training, checkpointing, and deterministic evaluation."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dqn_assignment.agent import DQNAgent
from dqn_assignment.config import DQNConfig, ExperimentConfig
from dqn_assignment.envs import ObservationStacker, adapt_observation, make_environment
from dqn_assignment.metrics import EpisodeRecord, read_episode_records, write_episode_records
from dqn_assignment.networks import AtariQNetwork, MLPQNetwork
from dqn_assignment.plotting import (
    plot_evaluation_histogram,
    plot_training,
    write_evaluation_summary,
)
from dqn_assignment.replay import ReplayBuffer
from dqn_assignment.schedules import linear_schedule


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Paths and final counters from a completed training invocation."""

    checkpoint_path: Path
    metrics_path: Path
    episodes_completed: int
    environment_steps: int


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Raw and aggregate results from checkpoint evaluation."""

    rewards: tuple[float, ...]
    mean_reward: float
    standard_deviation: float
    summary_path: Path
    histogram_path: Path


def resolve_device(requested: str) -> torch.device:
    """Resolve auto/cpu/cuda without silently falling back from requested CUDA."""

    normalized = requested.lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available to PyTorch")
    if normalized not in {"cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    return torch.device(normalized)


def set_global_seeds(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_network(observation_shape: tuple[int, ...], action_count: int) -> torch.nn.Module:
    if len(observation_shape) == 1:
        return MLPQNetwork(observation_shape[0], action_count)
    if len(observation_shape) == 3:
        return AtariQNetwork(observation_shape[0], action_count)
    raise ValueError(f"Unsupported observation shape: {observation_shape}")


def _save_checkpoint(
    path: Path,
    *,
    agent: DQNAgent,
    config: ExperimentConfig,
    episode: int,
    environment_steps: int,
    observation_shape: tuple[int, ...],
    action_count: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format_version": 1,
            "agent": agent.state_dict(),
            "config": {
                "environment_id": config.environment_id,
                "episodes": config.episodes,
                "max_steps_per_episode": config.max_steps_per_episode,
                "seed": config.seed,
                "expected_action_count": config.expected_action_count,
                "frame_stack": config.frame_stack,
                "output_dir": str(config.output_dir),
                "checkpoint_every": config.checkpoint_every,
                "evaluation_episodes": config.evaluation_episodes,
                "dqn": {name: getattr(config.dqn, name) for name in DQNConfig.__dataclass_fields__},
            },
            "episode": episode,
            "environment_steps": environment_steps,
            "observation_shape": observation_shape,
            "action_count": action_count,
        },
        temporary_path,
    )
    temporary_path.replace(path)


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    config_payload = checkpoint["config"]
    checkpoint["config"] = ExperimentConfig(
        environment_id=config_payload["environment_id"],
        episodes=int(config_payload["episodes"]),
        max_steps_per_episode=config_payload["max_steps_per_episode"],
        seed=int(config_payload["seed"]),
        expected_action_count=config_payload["expected_action_count"],
        frame_stack=int(config_payload["frame_stack"]),
        output_dir=Path(config_payload["output_dir"]),
        checkpoint_every=int(config_payload["checkpoint_every"]),
        evaluation_episodes=int(config_payload["evaluation_episodes"]),
        dqn=DQNConfig(**config_payload["dqn"]),
    )
    return checkpoint


def train(
    config: ExperimentConfig,
    *,
    device: str = "auto",
    resume_checkpoint: Path | None = None,
) -> TrainingResult:
    """Train one DQN run and generate all training plots."""

    selected_device = resolve_device(device)
    set_global_seeds(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    environment = make_environment(config)
    records: list[EpisodeRecord] = []
    environment_steps = 0
    start_episode = 0
    started_at = time.monotonic()
    checkpoint_path = config.output_dir / "checkpoints" / "final.pt"
    metrics_path = config.output_dir / "metrics" / "episodes.csv"

    try:
        raw_observation, _ = environment.reset(seed=config.seed)
        environment.action_space.seed(config.seed)
        stacker = ObservationStacker(config.frame_stack)
        observation = stacker.reset(adapt_observation(config.environment_id, raw_observation))
        action_count = int(environment.action_space.n)  # type: ignore[attr-defined]
        observation_shape = tuple(observation.shape)
        agent = DQNAgent(
            _build_network(observation_shape, action_count),
            None,
            action_count,
            config.dqn,
            selected_device,
            seed=config.seed,
        )
        replay = ReplayBuffer(config.dqn.replay_capacity, seed=config.seed)

        if resume_checkpoint is not None:
            checkpoint = _load_checkpoint(resume_checkpoint, selected_device)
            checkpoint_config: ExperimentConfig = checkpoint["config"]
            if checkpoint_config.environment_id != config.environment_id:
                raise ValueError("Resume checkpoint environment does not match configuration")
            if checkpoint_config.dqn != config.dqn:
                raise ValueError("Resume checkpoint DQN hyperparameters do not match configuration")
            if int(checkpoint["action_count"]) != action_count:
                raise ValueError("Resume checkpoint action count does not match environment")
            if tuple(checkpoint["observation_shape"]) != observation_shape:
                raise ValueError("Resume checkpoint observation shape does not match environment")
            agent.load_state_dict(checkpoint["agent"])
            start_episode = int(checkpoint["episode"])
            environment_steps = int(checkpoint["environment_steps"])
            if start_episode >= config.episodes:
                raise ValueError("Resume checkpoint already reached the requested episode count")
            if not metrics_path.exists():
                raise FileNotFoundError(
                    "Resume requires the original metrics/episodes.csv so plots retain "
                    "the complete training history"
                )
            records = read_episode_records(metrics_path)
            if len(records) < start_episode:
                raise ValueError("Resume metrics contain fewer episodes than the checkpoint")
            records = records[:start_episode]
            print(
                f"Resuming at episode {start_episode + 1} with a fresh replay buffer; "
                "optimization restarts after the buffer contains one batch."
            )

        starting_environment_steps = environment_steps

        for episode_index in range(start_episode, config.episodes):
            raw_observation, _ = environment.reset(seed=config.seed + episode_index)
            observation = stacker.reset(adapt_observation(config.environment_id, raw_observation))
            episode_reward = 0.0
            q_values: list[float] = []
            losses: list[float] = []
            td_errors: list[float] = []
            action_counts = np.zeros(action_count, dtype=np.int64)
            max_steps = config.max_steps_per_episode or 1_000_000_000
            final_epsilon = config.dqn.epsilon_start

            for step_index in range(max_steps):
                final_epsilon = linear_schedule(
                    config.dqn.epsilon_start,
                    config.dqn.epsilon_end,
                    config.dqn.epsilon_decay_steps,
                    environment_steps,
                )
                action, max_q = agent.select_action(observation, final_epsilon)
                next_raw, reward, terminated, truncated, _ = environment.step(action)
                next_observation = stacker.append(
                    adapt_observation(config.environment_id, next_raw)
                )
                artificial_truncation = step_index + 1 >= max_steps and not terminated
                episode_done = bool(terminated or truncated or artificial_truncation)

                # Only true MDP terminals suppress bootstrapping. TimeLimit truncation
                # retains the next-state value, which avoids a biased low target.
                replay.add(
                    observation,
                    action,
                    float(reward),
                    next_observation,
                    bool(terminated),
                )
                observation = next_observation
                environment_steps += 1
                episode_reward += float(reward)
                q_values.append(max_q)
                action_counts[action] += 1

                ready = (
                    environment_steps >= config.dqn.warmup_steps
                    and len(replay) >= config.dqn.batch_size
                    and environment_steps % config.dqn.train_frequency == 0
                )
                if ready:
                    optimization = agent.optimize(
                        replay.sample(config.dqn.batch_size, selected_device)
                    )
                    losses.append(optimization.loss)
                    td_errors.append(optimization.mean_absolute_td_error)
                if environment_steps % config.dqn.target_update_interval == 0:
                    agent.sync_target_network()
                if episode_done:
                    break

            record = EpisodeRecord(
                episode=episode_index + 1,
                reward=episode_reward,
                length=step_index + 1,
                mean_max_q=float(np.mean(q_values)),
                epsilon=final_epsilon,
                mean_loss=float(np.mean(losses)) if losses else 0.0,
                mean_absolute_td_error=float(np.mean(td_errors)) if td_errors else 0.0,
                action_counts=tuple(int(value) for value in action_counts),
            )
            records.append(record)
            write_episode_records(metrics_path, records)

            if (episode_index + 1) % 20 == 0 or episode_index + 1 == config.episodes:
                elapsed = max(time.monotonic() - started_at, 1e-9)
                completed_this_run = episode_index - start_episode + 1
                remaining = config.episodes - episode_index - 1
                eta_seconds = remaining * elapsed / completed_this_run
                recent_reward = float(np.mean([item.reward for item in records[-100:]]))
                print(
                    f"episode={episode_index + 1}/{config.episodes} "
                    f"reward={episode_reward:.1f} avg100={recent_reward:.1f} "
                    f"epsilon={final_epsilon:.4f} loss={record.mean_loss:.5f} "
                    f"steps_per_second="
                    f"{(environment_steps - starting_environment_steps) / elapsed:.1f} "
                    f"eta_minutes={eta_seconds / 60:.1f}",
                    flush=True,
                )

            if (episode_index + 1) % config.checkpoint_every == 0:
                periodic_path = (
                    config.output_dir / "checkpoints" / f"episode_{episode_index + 1:05d}.pt"
                )
                _save_checkpoint(
                    periodic_path,
                    agent=agent,
                    config=config,
                    episode=episode_index + 1,
                    environment_steps=environment_steps,
                    observation_shape=observation_shape,
                    action_count=action_count,
                )

        _save_checkpoint(
            checkpoint_path,
            agent=agent,
            config=config,
            episode=config.episodes,
            environment_steps=environment_steps,
            observation_shape=observation_shape,
            action_count=action_count,
        )
        plot_training(records, config.output_dir)
    finally:
        environment.close()

    return TrainingResult(
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
        episodes_completed=config.episodes,
        environment_steps=environment_steps,
    )


def evaluate_checkpoint(
    checkpoint_path: Path,
    *,
    episodes: int = 500,
    device: str = "auto",
    output_dir: Path | None = None,
) -> EvaluationResult:
    """Run greedy evaluation episodes and create the required histogram."""

    if episodes <= 0:
        raise ValueError("episodes must be positive")
    selected_device = resolve_device(device)
    checkpoint = _load_checkpoint(checkpoint_path, selected_device)
    config: ExperimentConfig = checkpoint["config"]
    environment = make_environment(config)
    destination = output_dir or checkpoint_path.parent.parent / "evaluation"
    rewards: list[float] = []

    try:
        observation_shape = tuple(checkpoint["observation_shape"])
        action_count = int(checkpoint["action_count"])
        agent = DQNAgent(
            _build_network(observation_shape, action_count),
            None,
            action_count,
            config.dqn,
            selected_device,
            seed=config.seed,
        )
        agent.load_state_dict(checkpoint["agent"])

        for episode_index in range(episodes):
            raw_observation, _ = environment.reset(seed=config.seed + 100_000 + episode_index)
            stacker = ObservationStacker(config.frame_stack)
            observation = stacker.reset(adapt_observation(config.environment_id, raw_observation))
            episode_reward = 0.0
            max_steps = config.max_steps_per_episode or 1_000_000_000
            for _ in range(max_steps):
                action, _ = agent.select_action(observation, epsilon=0.0)
                next_raw, reward, terminated, truncated, _ = environment.step(action)
                observation = stacker.append(adapt_observation(config.environment_id, next_raw))
                episode_reward += float(reward)
                if terminated or truncated:
                    break
            rewards.append(episode_reward)
    finally:
        environment.close()

    summary_path = write_evaluation_summary(destination / "summary.json", rewards)
    histogram_path = plot_evaluation_histogram(
        rewards,
        destination / "reward_histogram.png",
        title=f"{config.environment_id}: Greedy Policy Rewards ({episodes} Episodes)",
    )
    values = np.asarray(rewards, dtype=np.float64)
    return EvaluationResult(
        rewards=tuple(rewards),
        mean_reward=float(values.mean()),
        standard_deviation=float(values.std(ddof=0)),
        summary_path=summary_path,
        histogram_path=histogram_path,
    )
