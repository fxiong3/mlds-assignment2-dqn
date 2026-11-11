"""Validated experiment configuration and assignment presets."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DQNConfig:
    """Hyperparameters shared by the agent and replay loop."""

    gamma: float = 0.99
    learning_rate: float = 1e-4
    batch_size: int = 64
    replay_capacity: int = 50_000
    warmup_steps: int = 1_000
    train_frequency: int = 1
    target_update_interval: int = 1_000
    gradient_clip_norm: float = 10.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 100_000
    double_dqn: bool = False

    def __post_init__(self) -> None:
        positive_fields = {
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "replay_capacity": self.replay_capacity,
            "train_frequency": self.train_frequency,
            "target_update_interval": self.target_update_interval,
            "gradient_clip_norm": self.gradient_clip_norm,
            "epsilon_decay_steps": self.epsilon_decay_steps,
        }
        for name, value in positive_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps cannot be negative")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1")
        if not 0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0:
            raise ValueError("epsilon values must satisfy 0 <= end <= start <= 1")
        if self.replay_capacity < self.batch_size:
            raise ValueError("replay_capacity must be at least batch_size")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Complete configuration for one training run."""

    environment_id: str
    episodes: int = 500
    max_steps_per_episode: int | None = None
    seed: int = 42
    expected_action_count: int | None = None
    frame_stack: int = 1
    output_dir: Path = Path("runs/default")
    checkpoint_every: int = 100
    evaluation_episodes: int = 500
    dqn: DQNConfig = field(default_factory=DQNConfig)

    def __post_init__(self) -> None:
        if not self.environment_id:
            raise ValueError("environment_id cannot be empty")
        if self.episodes <= 0:
            raise ValueError("episodes must be positive")
        if self.max_steps_per_episode is not None and self.max_steps_per_episode <= 0:
            raise ValueError("max_steps_per_episode must be positive when provided")
        if self.checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be positive")
        if self.evaluation_episodes <= 0:
            raise ValueError("evaluation_episodes must be positive")
        if self.expected_action_count is not None and self.expected_action_count <= 0:
            raise ValueError("expected_action_count must be positive when provided")
        if self.frame_stack <= 0:
            raise ValueError("frame_stack must be positive")

    def with_overrides(
        self,
        *,
        episodes: int | None = None,
        seed: int | None = None,
        output_dir: Path | None = None,
        max_steps_per_episode: int | None = None,
    ) -> ExperimentConfig:
        """Return a validated copy with command-line overrides."""

        return replace(
            self,
            episodes=self.episodes if episodes is None else episodes,
            seed=self.seed if seed is None else seed,
            output_dir=self.output_dir if output_dir is None else output_dir,
            max_steps_per_episode=(
                self.max_steps_per_episode
                if max_steps_per_episode is None
                else max_steps_per_episode
            ),
        )


def preset_for(name: str) -> ExperimentConfig:
    """Return an assignment-aligned CartPole or MsPacman preset."""

    normalized = name.strip().lower().replace("_", "").replace("-", "")
    if normalized == "cartpole":
        return ExperimentConfig(
            environment_id="CartPole-v1",
            episodes=600,
            max_steps_per_episode=500,
            expected_action_count=2,
            output_dir=Path("runs/cartpole"),
            checkpoint_every=50,
            dqn=DQNConfig(
                gamma=0.95,
                learning_rate=1e-3,
                batch_size=128,
                replay_capacity=50_000,
                warmup_steps=1_000,
                train_frequency=1,
                target_update_interval=500,
                epsilon_start=1.0,
                epsilon_end=0.02,
                epsilon_decay_steps=25_000,
            ),
        )
    if normalized in {"mspacman", "pacman"}:
        return ExperimentConfig(
            environment_id="ALE/MsPacman-v5",
            episodes=6_000,
            expected_action_count=9,
            frame_stack=4,
            output_dir=Path("runs/mspacman"),
            checkpoint_every=100,
            dqn=DQNConfig(
                gamma=0.99,
                learning_rate=1e-4,
                batch_size=32,
                replay_capacity=50_000,
                warmup_steps=20_000,
                train_frequency=4,
                target_update_interval=10_000,
                epsilon_start=1.0,
                epsilon_end=0.05,
                epsilon_decay_steps=1_000_000,
                double_dqn=True,
            ),
        )
    raise ValueError(f"Unknown preset {name!r}; choose 'cartpole' or 'mspacman'")
