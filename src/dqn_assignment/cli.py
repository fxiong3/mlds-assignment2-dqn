"""Command-line interface for training, evaluation, plotting, and validation."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from dqn_assignment.config import ExperimentConfig, preset_for
from dqn_assignment.metrics import read_episode_records
from dqn_assignment.plotting import plot_training
from dqn_assignment.training import evaluate_checkpoint, train
from dqn_assignment.validation import run_validation


def build_parser() -> argparse.ArgumentParser:
    """Build the public command-line interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train CartPole or MsPacman DQN")
    train_parser.add_argument("--env", choices=["cartpole", "mspacman"], required=True)
    train_parser.add_argument("--episodes", type=int)
    train_parser.add_argument("--seed", type=int)
    train_parser.add_argument("--output", type=Path)
    train_parser.add_argument("--max-steps", type=int)
    train_parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    train_parser.add_argument("--learning-rate", type=float)
    train_parser.add_argument("--batch-size", type=int)
    train_parser.add_argument("--warmup-steps", type=int)
    train_parser.add_argument("--target-update-interval", type=int)
    train_parser.add_argument("--epsilon-decay-steps", type=int)
    train_parser.add_argument("--replay-capacity", type=int)
    train_parser.add_argument(
        "--resume",
        type=Path,
        help="Warm-resume model/optimizer state; replay memory starts empty",
    )

    evaluate_parser = subparsers.add_parser("evaluate", help="Run greedy checkpoint rollouts")
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    evaluate_parser.add_argument("--episodes", type=int, default=500)
    evaluate_parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    evaluate_parser.add_argument("--output", type=Path)

    plot_parser = subparsers.add_parser("plot", help="Regenerate plots from episode CSV")
    plot_parser.add_argument("--run-dir", type=Path, required=True)

    validation_parser = subparsers.add_parser(
        "validate", help="Compare a checkpoint against random actions and trace decisions"
    )
    validation_parser.add_argument("--checkpoint", type=Path, required=True)
    validation_parser.add_argument("--episodes", type=int, default=30)
    validation_parser.add_argument("--trace-steps", type=int, default=10)
    validation_parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    validation_parser.add_argument("--output", type=Path)
    return parser


def config_from_train_args(arguments: argparse.Namespace) -> ExperimentConfig:
    """Apply explicit CLI overrides to a validated assignment preset."""

    config = preset_for(arguments.env).with_overrides(
        episodes=arguments.episodes,
        seed=arguments.seed,
        output_dir=arguments.output,
        max_steps_per_episode=arguments.max_steps,
    )
    dqn_overrides = {
        "learning_rate": arguments.learning_rate,
        "batch_size": arguments.batch_size,
        "warmup_steps": arguments.warmup_steps,
        "target_update_interval": arguments.target_update_interval,
        "epsilon_decay_steps": arguments.epsilon_decay_steps,
        "replay_capacity": arguments.replay_capacity,
    }
    supplied = {name: value for name, value in dqn_overrides.items() if value is not None}
    return replace(config, dqn=replace(config.dqn, **supplied))


def main(argv: list[str] | None = None) -> int:
    """Execute one CLI command and print the essential result paths."""

    arguments = build_parser().parse_args(argv)
    if arguments.command == "train":
        result = train(
            config_from_train_args(arguments),
            device=arguments.device,
            resume_checkpoint=arguments.resume,
        )
        print(f"checkpoint={result.checkpoint_path}")
        print(f"metrics={result.metrics_path}")
        print(f"environment_steps={result.environment_steps}")
    elif arguments.command == "evaluate":
        result = evaluate_checkpoint(
            arguments.checkpoint,
            episodes=arguments.episodes,
            device=arguments.device,
            output_dir=arguments.output,
        )
        print(f"mean_reward={result.mean_reward:.6f}")
        print(f"standard_deviation={result.standard_deviation:.6f}")
        print(f"summary={result.summary_path}")
        print(f"histogram={result.histogram_path}")
    elif arguments.command == "plot":
        metrics_path = arguments.run_dir / "metrics" / "episodes.csv"
        paths = plot_training(read_episode_records(metrics_path), arguments.run_dir)
        for path in paths:
            print(path)
    elif arguments.command == "validate":
        summary_path = run_validation(
            arguments.checkpoint,
            episodes=arguments.episodes,
            trace_steps=arguments.trace_steps,
            device=arguments.device,
            output_dir=arguments.output,
        )
        print(f"validation_summary={summary_path}")
    else:  # pragma: no cover - argparse enforces a known subcommand
        raise AssertionError(f"Unhandled command: {arguments.command}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
