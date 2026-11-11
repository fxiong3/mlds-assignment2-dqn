import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn

from dqn_assignment.agent import DQNAgent, compute_td_targets
from dqn_assignment.config import DQNConfig
from dqn_assignment.replay import TransitionBatch


class ConstantQNetwork(nn.Module):
    def __init__(self, values: tuple[float, ...]) -> None:
        super().__init__()
        self.values = nn.Parameter(torch.tensor(values, dtype=torch.float32))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.values.unsqueeze(0).expand(observations.shape[0], -1)


def test_td_target_does_not_bootstrap_true_terminal() -> None:
    rewards = torch.tensor([[1.0], [1.0]])
    terminals = torch.tensor([[True], [False]])
    next_q = torch.tensor([[10.0], [10.0]])

    targets = compute_td_targets(rewards, terminals, next_q, gamma=0.9)

    assert torch.allclose(targets, torch.tensor([[1.0], [10.0]]))


def test_agent_syncs_target_network_on_creation() -> None:
    online = ConstantQNetwork((1.0, 2.0))
    target = ConstantQNetwork((-3.0, -4.0))

    agent = DQNAgent(online, target, action_count=2, config=DQNConfig(), device="cpu")

    assert torch.equal(agent.online_network.values, agent.target_network.values)
    assert not agent.target_network.training


def test_greedy_action_selects_largest_q_value() -> None:
    agent = DQNAgent(
        ConstantQNetwork((1.0, 5.0, 2.0)),
        ConstantQNetwork((0.0, 0.0, 0.0)),
        action_count=3,
        config=DQNConfig(),
        device="cpu",
        seed=2,
    )

    action, max_q = agent.select_action(np.array([0.0], dtype=np.float32), epsilon=0.0)

    assert action == 1
    assert max_q == pytest.approx(5.0)


def test_random_action_is_within_action_space() -> None:
    agent = DQNAgent(
        ConstantQNetwork((1.0, 5.0)),
        ConstantQNetwork((0.0, 0.0)),
        action_count=2,
        config=DQNConfig(),
        device="cpu",
        seed=2,
    )

    actions = [
        agent.select_action(np.array([0.0], dtype=np.float32), epsilon=1.0)[0] for _ in range(20)
    ]

    assert set(actions) == {0, 1}


def test_optimize_returns_loss_and_td_error() -> None:
    config = DQNConfig(gamma=0.9, learning_rate=1e-3)
    agent = DQNAgent(
        ConstantQNetwork((1.0, 2.0)),
        ConstantQNetwork((1.0, 2.0)),
        action_count=2,
        config=config,
        device="cpu",
    )
    batch = TransitionBatch(
        states=torch.zeros((2, 1)),
        actions=torch.tensor([[0], [1]]),
        rewards=torch.tensor([[1.0], [2.0]]),
        next_states=torch.zeros((2, 1)),
        terminals=torch.tensor([[False], [True]]),
    )

    result = agent.optimize(batch)

    assert result.loss >= 0.0
    assert result.mean_absolute_td_error >= 0.0
