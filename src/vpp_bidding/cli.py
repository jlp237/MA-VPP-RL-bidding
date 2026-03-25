"""Command-line interface for VPP Bidding."""

import argparse
import sys
from pathlib import Path

from vpp_bidding.domain.enums import Algorithm
from vpp_bidding.utils.logging import setup_logging


def _add_train_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("train", help="Train an RL agent")
    parser.add_argument("--config", type=Path, required=True, help="Path to TOML config file")
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=[a.value for a in Algorithm],
        default="PPO",
        help="RL algorithm to use (default: PPO)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=11140,
        help="Total training timesteps (default: 11140)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")


def _add_evaluate_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("evaluate", help="Evaluate a trained agent")
    parser.add_argument("--model", type=Path, required=True, help="Path to saved model")
    parser.add_argument("--config", type=Path, required=True, help="Path to TOML config file")
    parser.add_argument(
        "--episodes",
        type=int,
        default=70,
        help="Number of evaluation episodes (default: 70)",
    )


def _add_tune_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("tune", help="Run hyperparameter tuning")
    parser.add_argument("--config", type=Path, required=True, help="Path to TOML config file")
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=[a.value for a in Algorithm],
        default="PPO",
        help="RL algorithm to tune (default: PPO)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of Optuna trials (default: 100)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=21600,
        help="Timeout in seconds (default: 21600)",
    )


def _add_collect_data_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser("collect-data", help="Collect raw data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Output directory for raw data (default: data/raw)",
    )
    parser.add_argument("--start-date", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="End date (YYYY-MM-DD)")


def _run_train(args: argparse.Namespace) -> None:
    from vpp_bidding.training.train import train

    algorithm = Algorithm(args.algorithm)
    train(
        config_path=args.config,
        algorithm=algorithm,
        total_timesteps=args.timesteps,
        seed=args.seed,
    )


def _run_evaluate(args: argparse.Namespace) -> None:
    from vpp_bidding.training.evaluate import evaluate

    metrics = evaluate(
        model_path=args.model,
        config_path=args.config,
        episodes=args.episodes,
    )
    print("Evaluation Results:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


def _run_tune(args: argparse.Namespace) -> None:
    from vpp_bidding.training.tuning import tune

    algorithm = Algorithm(args.algorithm)
    tune(
        config_path=args.config,
        algorithm=algorithm,
        n_trials=args.n_trials,
        timeout=args.timeout,
    )


def _run_collect_data(args: argparse.Namespace) -> None:
    from vpp_bidding.data.collect import (
        fetch_fcr_tenders,
        fetch_renewable_profiles,
        fetch_wholesale_prices,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    fetch_fcr_tenders(output_dir)

    if args.start_date and args.end_date:
        fetch_wholesale_prices(output_dir, args.start_date, args.end_date)

    fetch_renewable_profiles(output_dir)


def main() -> None:
    """Entry point for the VPP Bidding CLI."""
    parser = argparse.ArgumentParser(
        prog="vpp-bidding",
        description="RL-based strategic bidding for Virtual Power Plants",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="WARNING",
        help="Logging level (default: WARNING)",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional log file path",
    )

    subparsers = parser.add_subparsers(dest="command")
    _add_train_parser(subparsers)
    _add_evaluate_parser(subparsers)
    _add_tune_parser(subparsers)
    _add_collect_data_parser(subparsers)

    args = parser.parse_args()

    setup_logging(level=args.log_level, log_file=args.log_file)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    dispatch = {
        "train": _run_train,
        "evaluate": _run_evaluate,
        "tune": _run_tune,
        "collect-data": _run_collect_data,
    }

    dispatch[args.command](args)
