from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gymnasium")
torch = pytest.importorskip("torch")

from dqn_assignment.config import DQNConfig, ExperimentConfig
from dqn_assignment.metrics import read_episode_records
from dqn_assignment.training import _replay_reward, evaluate_checkpoint, resolve_device, train


def tiny_cartpole_config(output_dir: Path) -> ExperimentConfig:
    return ExperimentConfig(
        environment_id="CartPole-v1",
        episodes=2,
        max_steps_per_episode=5,
        expected_action_count=2,
        output_dir=output_dir,
        checkpoint_every=1,
        evaluation_episodes=2,
        dqn=DQNConfig(
            gamma=0.95,
            learning_rate=1e-3,
            batch_size=2,
            replay_capacity=20,
            warmup_steps=0,
            train_frequency=1,
            target_update_interval=2,
            epsilon_decay_steps=5,
        ),
    )


def test_resolve_device_accepts_cpu_and_rejects_missing_cuda() -> None:
    assert resolve_device("cpu") == torch.device("cpu")
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA"):
            resolve_device("cuda")


@pytest.mark.parametrize(
    ("environment_id", "reward", "terminated", "next_observation", "expected"),
    [
        ("CartPole-v1", 1.0, True, [0.0, 0.0, 0.0, 0.0], -1.0),
        ("CartPole-v1", 1.0, False, [0.0, 0.0, 0.0, 0.0], 1.0),
        ("CartPole-v1", 1.0, False, [1.2, 0.0, 0.10471976, 0.0], 0.5),
        ("CartPole-v1", 1.0, False, [-1.2, 0.0, -0.10471976, 0.0], 0.5),
        ("CartPole-v1", 1.0, False, [2.4, 0.0, 0.20943951, 0.0], 0.0),
        ("CartPole-v1", 1.0, False, [4.8, 0.0, 0.41887902, 0.0], 0.0),
        ("ALE/MsPacman-v5", 10.0, True, [0.0], 10.0),
    ],
)
def test_replay_reward_shapes_only_cartpole(
    environment_id: str,
    reward: float,
    terminated: bool,
    next_observation: list[float],
    expected: float,
) -> None:
    assert _replay_reward(
        environment_id,
        reward,
        terminated,
        np.asarray(next_observation, dtype=np.float32),
    ) == pytest.approx(expected)


def test_short_training_writes_metrics_checkpoint_and_plots(tmp_path: Path) -> None:
    config = tiny_cartpole_config(tmp_path / "run")

    result = train(config, device="cpu")

    assert result.checkpoint_path.exists()
    assert result.metrics_path.exists()
    records = read_episode_records(result.metrics_path)
    assert len(records) == 2
    assert all(record.reward == record.length for record in records)
    assert (config.output_dir / "plots" / "episode_rewards.png").exists()
    assert (config.output_dir / "plots" / "max_q_values.png").exists()
    assert (config.output_dir / "plots" / "action_frequency_heatmap.png").exists()


def test_checkpoint_can_be_evaluated(tmp_path: Path) -> None:
    config = tiny_cartpole_config(tmp_path / "run")
    training_result = train(config, device="cpu")

    evaluation = evaluate_checkpoint(
        training_result.checkpoint_path,
        episodes=2,
        device="cpu",
        output_dir=tmp_path / "evaluation",
    )

    assert len(evaluation.rewards) == 2
    assert evaluation.summary_path.exists()
    assert evaluation.histogram_path.exists()
    assert evaluation.standard_deviation >= 0.0


def test_checkpoint_can_warm_resume_with_fresh_replay(tmp_path: Path) -> None:
    first_config = replace(tiny_cartpole_config(tmp_path / "run"), episodes=1)
    first_result = train(first_config, device="cpu")
    continued_config = replace(first_config, episodes=2)

    continued_result = train(
        continued_config,
        device="cpu",
        resume_checkpoint=first_result.checkpoint_path,
    )

    records = read_episode_records(continued_result.metrics_path)
    assert [record.episode for record in records] == [1, 2]


def test_configuration_can_be_replaced_for_ablation(tmp_path: Path) -> None:
    config = tiny_cartpole_config(tmp_path)

    changed = replace(config, dqn=replace(config.dqn, target_update_interval=17))

    assert changed.dqn.target_update_interval == 17
