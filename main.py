import argparse
import sys
import warnings

from optimization.tune import tune
from prediction.train import train
from simulation.simulate import simulate


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore", category=UserWarning)

    # Set up command-line argument parsing
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    simulate_parser = subparsers.add_parser("simulate", help="Simulate seasons")
    _ = subparsers.add_parser("tune", help="Tune hyperparameters")
    _ = subparsers.add_parser("train", help="Train models")

    simulate_parser.add_argument(
        "--season",
        type=str,
        choices=["2021-22", "2022-23", "2023-24", "2024-25"],
        help="Season to simulate",
    )
    simulate_parser.add_argument(
        "--log", action="store_true", help="Log simulation details"
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
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
