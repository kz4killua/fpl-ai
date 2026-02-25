import argparse
import random
import subprocess

import numpy as np
import polars as pl

from game.run import run
from loaders.constants import DATA_REPO_PATH
from optimization.tune import tune
from prediction.train import train
from simulation.simulate import simulate


def main():
    set_seed(42)

    # Set up command-line argument parsing
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    run_parser = subparsers.add_parser(
        "run", help="Run optimization on a live FPL team"
    )
    run_parser.add_argument(
        "--season",
        type=int,
        required=True,
        help="The current season",
    )
    run_parser.add_argument(
        "--next-gameweek",
        type=int,
        required=True,
        help="The next gameweek",
    )
    run_parser.add_argument(
        "--wildcard-gameweeks",
        type=int,
        nargs="*",
        default=[],
        help="Gameweeks to activate wildcards",
    )

    simulate_parser = subparsers.add_parser("simulate", help="Simulate seasons")
    simulate_parser.add_argument(
        "--season",
        type=int,
        choices=[2022, 2023, 2024],
        help="Season to simulate",
    )
    simulate_parser.add_argument(
        "--log", action="store_true", help="Log simulation details"
    )

    subparsers.add_parser("tune", help="Tune hyperparameters")
    subparsers.add_parser("train", help="Train models")

    # Ensure the data repository is up to date
    subprocess.run(
        ["git", "pull"],
        cwd=DATA_REPO_PATH,
        capture_output=True,
        text=True,
        check=True,
    )

    # Execute the appropriate command based on user input
    args = parser.parse_args()
    if args.command == "tune":
        tune()
    elif args.command == "simulate":
        points = simulate(args.season, [], log=args.log)
        print(f"{args.season}: {points} points")
    elif args.command == "train":
        train()
        print("Models trained successfully.")
    elif args.command == "run":
        run(args.season, args.next_gameweek, args.wildcard_gameweeks)
    else:
        parser.print_help()


def set_seed(seed: int):
    """Set global random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    pl.set_random_seed(seed)


if __name__ == "__main__":
    main()
