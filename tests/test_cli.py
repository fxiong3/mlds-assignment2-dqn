from pathlib import Path

from dqn_assignment.cli import build_parser, config_from_train_args


def test_train_cli_applies_reproducible_hyperparameter_overrides(tmp_path: Path) -> None:
    parser = build_parser()
    arguments = parser.parse_args(
        [
            "train",
            "--env",
            "cartpole",
            "--episodes",
            "12",
            "--seed",
            "9",
            "--output",
            str(tmp_path),
            "--target-update-interval",
            "17",
            "--learning-rate",
            "0.0003",
        ]
    )

    config = config_from_train_args(arguments)

    assert config.episodes == 12
    assert config.seed == 9
    assert config.output_dir == tmp_path
    assert config.dqn.target_update_interval == 17
    assert config.dqn.learning_rate == 0.0003


def test_evaluate_cli_defaults_to_assignment_rollout_count() -> None:
    parser = build_parser()

    arguments = parser.parse_args(["evaluate", "--checkpoint", "model.pt"])

    assert arguments.episodes == 500
