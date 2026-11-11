from pathlib import Path

import numpy as np
import pytest

from dqn_assignment.metrics import (
    EpisodeRecord,
    moving_average,
    read_episode_records,
    write_episode_records,
)


def test_moving_average_uses_available_prefix_before_full_window() -> None:
    result = moving_average([1.0, 2.0, 3.0, 4.0], window=3)

    assert np.allclose(result, [1.0, 1.5, 2.0, 3.0])


def test_moving_average_rejects_nonpositive_window() -> None:
    with pytest.raises(ValueError, match="positive"):
        moving_average([1.0], window=0)


def test_episode_records_round_trip_csv(tmp_path: Path) -> None:
    records = [
        EpisodeRecord(
            episode=1,
            reward=10.0,
            length=10,
            mean_max_q=1.5,
            epsilon=0.5,
            mean_loss=0.1,
            mean_absolute_td_error=0.2,
            action_counts=(4, 6),
        )
    ]
    path = tmp_path / "episodes.csv"

    write_episode_records(path, records)
    loaded = read_episode_records(path)

    assert loaded == records
