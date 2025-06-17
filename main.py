import argparse

from optimization.tune import tune
from simulation.simulate import simulate


def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    simulate_parser = subparsers.add_parser("simulate", help="Simulate seasons")
    _ = subparsers.add_parser("tune", help="Tune hyperparameters")
    simulate_parser.add_argument(
        "--season",
        type=str,
        choices=["2021-22", "2022-23", "2023-24"],
        help="Season to simulate",
    )
    simulate_parser.add_argument(
        "--log", action="store_true", help="Log simulation details"
    )
    args = parser.parse_args()

    # Execute the appropriate command based on user input
    if args.command == "tune":
        tune()
    elif args.command == "simulate":
        # Simulate the season
        points = simulate(args.season, [], log=args.log)
        print(f"{args.season}: {points} points")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
