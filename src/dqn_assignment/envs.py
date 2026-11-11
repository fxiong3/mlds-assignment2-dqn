"""Gymnasium environment construction and observation adaptation."""

from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np

from dqn_assignment.config import ExperimentConfig
from dqn_assignment.preprocessing import preprocess_mspacman


class ObservationStacker:
    """Maintain recent channel-first frames so Atari state includes motion."""

    def __init__(self, stack_size: int) -> None:
        if stack_size <= 0:
            raise ValueError("stack_size must be positive")
        self.stack_size = stack_size
        self._frames: deque[np.ndarray] = deque(maxlen=stack_size)

    def reset(self, observation: np.ndarray) -> np.ndarray:
        """Start an episode by repeating its first observation."""

        self._frames.clear()
        for _ in range(self.stack_size):
            self._frames.append(observation.copy())
        return self._stack()

    def append(self, observation: np.ndarray) -> np.ndarray:
        """Append one observation and return a copy of the current stack."""

        if not self._frames:
            raise RuntimeError("reset must be called before append")
        if observation.shape != self._frames[0].shape:
            raise ValueError("observation shape changed within an episode")
        self._frames.append(observation.copy())
        return self._stack()

    def _stack(self) -> np.ndarray:
        if self.stack_size == 1:
            return self._frames[0].copy()
        return np.concatenate(tuple(self._frames), axis=0)


def is_mspacman(environment_id: str) -> bool:
    """Return whether an environment ID denotes the assignment Atari task."""

    return "mspacman" in environment_id.lower()


def adapt_observation(environment_id: str, observation: Any) -> np.ndarray:
    """Convert raw Gymnasium observations into replay-ready arrays."""

    array = np.asarray(observation)
    if is_mspacman(environment_id):
        return preprocess_mspacman(array)
    return array.astype(np.float32, copy=False)


def make_environment(
    config: ExperimentConfig,
    *,
    render_mode: str | None = None,
) -> gym.Env:
    """Create a configured environment and validate its discrete action space."""

    if is_mspacman(config.environment_id):
        try:
            import ale_py

            gym.register_envs(ale_py)
        except ImportError as exc:
            raise RuntimeError(
                "MsPacman requires ale-py. Install the project dependencies and Atari ROMs."
            ) from exc
    try:
        environment = gym.make(config.environment_id, render_mode=render_mode)
    except gym.error.Error as exc:
        if is_mspacman(config.environment_id):
            raise RuntimeError(
                "Could not create ALE/MsPacman-v5. Install ale-py and run "
                "`AutoROM --accept-license` in the active environment."
            ) from exc
        raise
    if not isinstance(environment.action_space, gym.spaces.Discrete):
        environment.close()
        raise TypeError("DQN implementation requires a discrete action space")
    if (
        config.expected_action_count is not None
        and environment.action_space.n != config.expected_action_count
    ):
        actual = environment.action_space.n
        environment.close()
        raise RuntimeError(
            f"Expected {config.expected_action_count} actions for {config.environment_id}, "
            f"but environment exposes {actual}"
        )
    return environment
