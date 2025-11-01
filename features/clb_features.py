import polars as pl

from features.toa_features import estimate_goals
from loaders.utils import force_dataframe


def compute_clb_features(matches: pl.DataFrame):
    matches = force_dataframe(matches)

    # Calculate expected win probabilities for both sides based on clubelo ratings
    team_h_elo_win_probability = calculate_elo_win_probability(
        pl.col("team_h_clb_elo"), pl.col("team_a_clb_elo")
    ).alias("team_h_clb_win_prob")
    team_a_elo_win_probability = (1 - team_h_elo_win_probability).alias(
        "team_a_clb_win_prob"
    )
    matches = matches.with_columns(
        team_h_elo_win_probability, team_a_elo_win_probability
    )

    # Calculate expected goals for both sides based on win probabilities
    predictions = []
    for match in matches.to_dicts():
        target_probs = [
            {
                "type": "h2h",
                "data": [
                    {"side": "home", "prob": match["team_h_clb_win_prob"]},
                    {"side": "away", "prob": match["team_a_clb_win_prob"]},
                ],
            }
        ]

        result = estimate_goals(target_probs, max_goals=10)
        if result.success:
            predicted_home_goals, predicted_away_goals = result.x
        else:
            predicted_home_goals = predicted_away_goals = None

        row = {
            "season": match["season"],
            "fixture_id": match["fixture_id"],
            "team_h_clb_expected_goals": predicted_home_goals,
            "team_a_clb_expected_goals": predicted_away_goals,
        }
        predictions.append(row)

    clb_predictions = pl.DataFrame(predictions)

    # Merge predictions back into matches DataFrame
    matches = matches.join(
        clb_predictions.select(
            "season",
            "fixture_id",
            "team_h_clb_expected_goals",
            "team_a_clb_expected_goals",
        ),
        on=["season", "fixture_id"],
        how="left",
    )

    return matches.lazy()


def calculate_elo_win_probability(team_h_elo: pl.Expr, team_a_elo: pl.Expr) -> pl.Expr:
    """Calculate the win probability for the home team based on clubelo.com ratings."""
    return 1 / (10 ** ((team_a_elo - team_h_elo) / 400) + 1)
