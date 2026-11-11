import numpy as np
import pytest

from dqn_assignment.preprocessing import prepare_observations, preprocess_mspacman


def test_preprocess_mspacman_matches_assignment_shape_and_dtype() -> None:
    observation = np.zeros((210, 160, 3), dtype=np.uint8)

    processed = preprocess_mspacman(observation)

    assert processed.shape == (1, 88, 80)
    assert processed.dtype == np.int8
    assert np.all(processed == -128)


def test_preprocess_mspacman_removes_pacman_color() -> None:
    observation = np.zeros((210, 160, 3), dtype=np.uint8)
    observation[1, 0] = np.array([210, 164, 74], dtype=np.uint8)

    processed = preprocess_mspacman(observation)

    assert processed[0, 0, 0] == -128


def test_preprocess_mspacman_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="210, 160, 3"):
        preprocess_mspacman(np.zeros((88, 80), dtype=np.uint8))


def test_prepare_observations_normalizes_int8_images() -> None:
    torch = pytest.importorskip("torch")
    observations = torch.tensor([[[[-128, 0, 127]]]], dtype=torch.int8)

    prepared = prepare_observations(observations)

    assert prepared.dtype == torch.float32
    assert torch.allclose(
        prepared,
        torch.tensor([[[[-1.0, 0.0, 127 / 128]]]], dtype=torch.float32),
    )


def test_prepare_observations_preserves_vector_scale() -> None:
    torch = pytest.importorskip("torch")
    observations = torch.tensor([[1.5, -2.0]], dtype=torch.float32)

    prepared = prepare_observations(observations)

    assert torch.equal(prepared, observations)
