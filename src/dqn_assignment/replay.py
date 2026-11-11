"""A fixed-size, dtype-preserving replay buffer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class TransitionBatch:
    """Tensor batch consumed by DQN optimization."""

    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    terminals: torch.Tensor


class ReplayBuffer:
    """Preallocated circular replay buffer with lazy shape initialization."""

    def __init__(self, capacity: int, seed: int = 0) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._rng = np.random.default_rng(seed)
        self._states: np.ndarray | None = None
        self._next_states: np.ndarray | None = None
        self._actions = np.empty(capacity, dtype=np.int64)
        self._rewards = np.empty(capacity, dtype=np.float32)
        self._terminals = np.empty(capacity, dtype=np.bool_)
        self._position = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    @property
    def storage_dtype(self) -> np.dtype | None:
        """Return observation storage dtype once the first transition is added."""

        return None if self._states is None else self._states.dtype

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminal: bool,
    ) -> None:
        """Store one transition, overwriting the oldest item when full."""

        state_array = np.asarray(state)
        next_state_array = np.asarray(next_state)
        if state_array.shape != next_state_array.shape:
            raise ValueError("state and next_state must have the same shape")
        if self._states is None:
            observation_shape = (self.capacity, *state_array.shape)
            self._states = np.empty(observation_shape, dtype=state_array.dtype)
            self._next_states = np.empty(observation_shape, dtype=state_array.dtype)
        assert self._next_states is not None
        if state_array.shape != self._states.shape[1:]:
            raise ValueError(
                f"state shape changed from {self._states.shape[1:]} to {state_array.shape}"
            )

        self._states[self._position] = state_array
        self._next_states[self._position] = next_state_array
        self._actions[self._position] = action
        self._rewards[self._position] = reward
        self._terminals[self._position] = terminal
        self._position = (self._position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device: str | torch.device) -> TransitionBatch:
        """Sample transitions uniformly without replacement."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self._size < batch_size:
            raise ValueError(
                f"Cannot sample {batch_size} transitions from a buffer with "
                f"{self._size} transitions"
            )
        assert self._states is not None and self._next_states is not None
        indices = self._rng.choice(self._size, size=batch_size, replace=False)
        return TransitionBatch(
            states=torch.as_tensor(self._states[indices], device=device),
            actions=torch.as_tensor(self._actions[indices], device=device).unsqueeze(1),
            rewards=torch.as_tensor(self._rewards[indices], device=device).unsqueeze(1),
            next_states=torch.as_tensor(self._next_states[indices], device=device),
            terminals=torch.as_tensor(self._terminals[indices], device=device).unsqueeze(1),
        )
