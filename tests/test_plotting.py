import warnings
from pathlib import Path

from dqn_assignment.metrics import EpisodeRecord
from dqn_assignment.plotting import plot_training


def test_single_episode_plot_has_no_identical_axis_warning(tmp_path: Path) -> None:
    record = EpisodeRecord(
        episode=1,
        reward=1.0,
        length=1,
        mean_max_q=0.5,
        epsilon=1.0,
        mean_loss=0.0,
        mean_absolute_td_error=0.0,
        action_counts=(1, 0),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        paths = plot_training([record], tmp_path)

    assert all(path.exists() for path in paths)
