"""Deep Q-Network action selection and optimization."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from dqn_assignment.config import DQNConfig
from dqn_assignment.preprocessing import prepare_observations
from dqn_assignment.replay import TransitionBatch


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    """Scalar diagnostics for one gradient update."""

    loss: float
    mean_absolute_td_error: float


def compute_td_targets(
    rewards: torch.Tensor,
    terminals: torch.Tensor,
    next_q_values: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Compute Bellman targets, bootstrapping through time-limit truncations only."""

    nonterminal = (~terminals.to(torch.bool)).to(rewards.dtype)
    return rewards + gamma * nonterminal * next_q_values


class DQNAgent:
    """Own online/target networks, optimizer, and epsilon-greedy policy."""

    def __init__(
        self,
        online_network: nn.Module,
        target_network: nn.Module | None,
        action_count: int,
        config: DQNConfig,
        device: str | torch.device,
        seed: int = 0,
    ) -> None:
        if action_count <= 0:
            raise ValueError("action_count must be positive")
        self.device = torch.device(device)
        self.online_network = online_network.to(self.device)
        self.target_network = (
            copy.deepcopy(online_network) if target_network is None else target_network
        ).to(self.device)
        self.action_count = action_count
        self.config = config
        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=config.learning_rate)
        self._rng = np.random.default_rng(seed)
        self.sync_target_network()

    def sync_target_network(self) -> None:
        """Hard-copy online parameters into the frozen target network."""

        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()
        for parameter in self.target_network.parameters():
            parameter.requires_grad_(False)

    def select_action(self, state: np.ndarray, epsilon: float) -> tuple[int, float]:
        """Choose an epsilon-greedy action and return the state's maximum Q-value."""

        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between 0 and 1")
        state_tensor = torch.as_tensor(state, device=self.device).unsqueeze(0)
        self.online_network.eval()
        with torch.no_grad():
            q_values = self.online_network(prepare_observations(state_tensor))
        max_q = float(q_values.max().item())
        if self._rng.random() < epsilon:
            return int(self._rng.integers(self.action_count)), max_q
        return int(q_values.argmax(dim=1).item()), max_q

    def optimize(self, batch: TransitionBatch) -> OptimizationResult:
        """Perform one Huber-loss DQN update."""

        self.online_network.train()
        states = prepare_observations(batch.states)
        next_states = prepare_observations(batch.next_states)
        predicted_q = self.online_network(states).gather(1, batch.actions.long())

        with torch.no_grad():
            if self.config.double_dqn:
                next_actions = self.online_network(next_states).argmax(dim=1, keepdim=True)
                next_q = self.target_network(next_states).gather(1, next_actions)
            else:
                next_q = self.target_network(next_states).max(dim=1, keepdim=True).values
            targets = compute_td_targets(
                batch.rewards,
                batch.terminals,
                next_q,
                gamma=self.config.gamma,
            )

        td_errors = targets - predicted_q
        loss = nn.functional.smooth_l1_loss(predicted_q, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_network.parameters(), self.config.gradient_clip_norm)
        self.optimizer.step()
        return OptimizationResult(
            loss=float(loss.item()),
            mean_absolute_td_error=float(td_errors.abs().mean().item()),
        )

    def state_dict(self) -> dict[str, object]:
        """Serialize all training state needed to resume."""

        return {
            "online_network": self.online_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore a checkpoint produced by :meth:`state_dict`."""

        self.online_network.load_state_dict(state["online_network"])  # type: ignore[arg-type]
        self.target_network.load_state_dict(state["target_network"])  # type: ignore[arg-type]
        self.optimizer.load_state_dict(state["optimizer"])  # type: ignore[arg-type]
        self.target_network.eval()
