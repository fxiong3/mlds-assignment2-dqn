import numpy as np
import pytest

from dqn_assignment.replay import ReplayBuffer


def make_transition(value: int) -> tuple[np.ndarray, int, float, np.ndarray, bool]:
    state = np.array([value, value + 1], dtype=np.float32)
    return state, value % 2, float(value), state + 1, value % 3 == 0


def test_replay_rejects_sampling_before_batch_is_available() -> None:
    buffer = ReplayBuffer(capacity=4, seed=7)
    buffer.add(*make_transition(1))

    with pytest.raises(ValueError, match="2 transitions"):
        buffer.sample(batch_size=2, device="cpu")


def test_replay_wraps_at_capacity_and_returns_tensor_batch() -> None:
    torch = pytest.importorskip("torch")
    buffer = ReplayBuffer(capacity=3, seed=7)
    for value in range(5):
        buffer.add(*make_transition(value))

    batch = buffer.sample(batch_size=3, device="cpu")

    assert len(buffer) == 3
    assert batch.states.shape == (3, 2)
    assert batch.actions.shape == (3, 1)
    assert batch.rewards.shape == (3, 1)
    assert batch.terminals.shape == (3, 1)
    assert batch.states.dtype == torch.float32
    assert set(batch.rewards.flatten().tolist()) == {2.0, 3.0, 4.0}


def test_replay_preserves_int8_image_storage() -> None:
    buffer = ReplayBuffer(capacity=2, seed=0)
    state = np.full((1, 88, 80), -128, dtype=np.int8)
    buffer.add(state, 0, 1.0, state, False)

    assert buffer.storage_dtype == np.dtype(np.int8)


@pytest.mark.parametrize("capacity", [0, -1])
def test_replay_requires_positive_capacity(capacity: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        ReplayBuffer(capacity=capacity)
