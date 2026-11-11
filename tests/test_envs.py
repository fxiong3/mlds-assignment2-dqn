import numpy as np
import pytest

from dqn_assignment.config import preset_for
from dqn_assignment.envs import ObservationStacker, adapt_observation, make_environment


def test_cartpole_observation_is_float32_vector() -> None:
    observation = adapt_observation("CartPole-v1", np.array([1, 2, 3, 4], dtype=np.float64))

    assert observation.shape == (4,)
    assert observation.dtype == np.float32


def test_mspacman_observation_uses_assignment_preprocessing() -> None:
    observation = adapt_observation("ALE/MsPacman-v5", np.zeros((210, 160, 3), dtype=np.uint8))

    assert observation.shape == (1, 88, 80)
    assert observation.dtype == np.int8


def test_observation_stacker_repeats_initial_frame_then_tracks_motion() -> None:
    stacker = ObservationStacker(stack_size=4)
    first = np.zeros((1, 2, 2), dtype=np.int8)
    second = np.ones((1, 2, 2), dtype=np.int8)

    initial_stack = stacker.reset(first)
    next_stack = stacker.append(second)

    assert initial_stack.shape == (4, 2, 2)
    assert np.all(initial_stack == 0)
    assert np.all(next_stack[:3] == 0)
    assert np.all(next_stack[3] == 1)


def test_make_cartpole_environment_has_expected_actions() -> None:
    pytest.importorskip("gymnasium")
    environment = make_environment(preset_for("cartpole"))
    try:
        assert environment.action_space.n == 2
    finally:
        environment.close()
