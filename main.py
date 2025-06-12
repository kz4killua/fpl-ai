import argparse
import random

import numpy as np

from optimization.tune import tune
from simulation.simulate import simulate


def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    simulate_parser = subparsers.add_parser("simulate", help="Simulate seasons")
    _ = subparsers.add_parser("tune", help="Tune hyperparameters")
    simulate_parser.add_argument(
        "--seasons",
        nargs="+",
        default=["2021-22", "2022-23", "2023-24"],
        help="Seasons to simulate",
    )
    simulate_parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="Random seeds for simulation",
    )
    simulate_parser.add_argument(
        "--log", action="store_true", help="Log simulation details"
    )
    args = parser.parse_args()

    # Execute the appropriate command based on user input
    if args.command == "tune":
        tune()
    elif args.command == "simulate":
        # Simulate each season with the specified random seeds
        for season in args.seasons:
            results = []
            for seed in args.seeds:
                # Pick wildcard gameweeks at random
                random.seed(seed)
                wildcard_gameweeks = [random.randint(3, 19), random.randint(20, 36)]
                # Simulate the season
                points = simulate(season, wildcard_gameweeks, log=args.log)
                results.append(points)
            # Calculate the average points across all seeds
            mean = np.mean(results)
            std = np.std(results)
            print(f"{season}: {mean:.2f} ± {std:.2f} points")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
