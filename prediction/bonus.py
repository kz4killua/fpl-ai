import numpy as np
import polars as pl

STD_DEV_BPS = 7.5
NUM_SIMULATIONS = 1_000
RANDOM_STATE = 42


def predict_bonus(X: pl.DataFrame) -> np.ndarray:
    """Awards bonus points for each player in each fixture based on predicted BPS."""
    predicted_bonus = np.zeros(len(X), dtype=float)

    # Run simulations for each fixture
    for season, fixture in (
        # The sort is necessary to ensure consistent ordering
        X.select(["season", "fixture"]).unique().sort(["season", "fixture"]).to_numpy()
    ):
        mask = (X["season"] == season) & (X["fixture"] == fixture)
        mask = mask.to_numpy()

        # Predict bonus points for players in the fixture
        group = X.filter(mask)["predicted_bps"].to_numpy()
        predicted_bonus[mask] = simulate_bonus(
            group,
            STD_DEV_BPS,
            NUM_SIMULATIONS,
        )

    return predicted_bonus


def simulate_bonus(predicted_bps: np.ndarray, std_dev_bps: float, n_simulations: int):
    """Run a Monte Carlo simulation to predict bonus for players in a single fixture"""

    n_players = predicted_bps.shape[0]

    # Set random seed for reproducibility
    np.random.seed(RANDOM_STATE)

    # Generate random BPS values for each simulation
    samples = np.random.normal(
        loc=predicted_bps, scale=std_dev_bps, size=(n_simulations, n_players)
    )

    # Get the indices of the top 3 players in each simulation
    ranks = np.argsort(-samples, axis=1)[:, :3]

    # Count how many times each player is in the top 3
    count_1st = np.bincount(ranks[:, 0], minlength=n_players)
    count_2nd = np.bincount(ranks[:, 1], minlength=n_players)
    count_3rd = np.bincount(ranks[:, 2], minlength=n_players)

    # Convert to probabilities
    p_1st = count_1st / n_simulations
    p_2nd = count_2nd / n_simulations
    p_3rd = count_3rd / n_simulations

    return p_1st * 3 + p_2nd * 2 + p_3rd * 1
