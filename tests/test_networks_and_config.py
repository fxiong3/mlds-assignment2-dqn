import pytest

torch = pytest.importorskip("torch")

from dqn_assignment.config import DQNConfig, ExperimentConfig, preset_for
from dqn_assignment.networks import AtariQNetwork, MLPQNetwork
from dqn_assignment.schedules import linear_schedule


def test_mlp_network_has_expected_output_shape() -> None:
    network = MLPQNetwork(input_dim=4, action_count=2)

    output = network(torch.zeros((3, 4)))

    assert output.shape == (3, 2)


def test_atari_network_has_expected_output_shape() -> None:
    network = AtariQNetwork(input_channels=1, action_count=9)

    output = network(torch.zeros((2, 1, 88, 80)))

    assert output.shape == (2, 9)


@pytest.mark.parametrize(
    ("step", "expected"),
    [(-1, 1.0), (0, 1.0), (5, 0.55), (10, 0.1), (100, 0.1)],
)
def test_linear_schedule_clamps_to_endpoints(step: int, expected: float) -> None:
    assert linear_schedule(1.0, 0.1, duration=10, step=step) == pytest.approx(expected)


def test_linear_schedule_requires_positive_duration() -> None:
    with pytest.raises(ValueError, match="positive"):
        linear_schedule(1.0, 0.1, duration=0, step=1)


def test_cartpole_preset_uses_assignment_discount_factor() -> None:
    preset = preset_for("cartpole")

    assert preset.environment_id == "CartPole-v1"
    assert preset.dqn.gamma == pytest.approx(0.95)


def test_mspacman_preset_uses_assignment_discount_factor() -> None:
    preset = preset_for("mspacman")

    assert preset.environment_id == "ALE/MsPacman-v5"
    assert preset.dqn.gamma == pytest.approx(0.99)
    assert preset.expected_action_count == 9
    assert preset.frame_stack == 4


def test_invalid_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        DQNConfig(batch_size=0)
    with pytest.raises(ValueError, match="episodes"):
        ExperimentConfig(environment_id="CartPole-v1", episodes=0)
