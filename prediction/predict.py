from pathlib import Path

import polars as pl

from prediction.model import PredictionModel

OUTPUT_DIR = Path("output")


def make_predictions(
    model: PredictionModel, players: pl.DataFrame, matches: pl.DataFrame
) -> pl.DataFrame:
    return model.predict(players, matches, return_dataframe=True)


def aggregate_predictions(df: pl.DataFrame):
    """Group predictions by season, gameweek, and element, summing total points."""
    df = df.select("season", "gameweek", "element", "predicted_total_points")
    df = df.rename({"predicted_total_points": "total_points"})
    df = df.group_by(["season", "gameweek", "element"]).agg(
        pl.col("total_points").sum(),
    )
    return df


def save_predictions(
    predictions: pl.DataFrame, static_elements: pl.DataFrame, static_teams: pl.DataFrame
):
    """Save player and team predictions to CSV files."""
    # Add player and team names to predictions
    df = predictions.join(
        static_elements.select(
            pl.col("season"),
            pl.col("id").alias("element"),
            pl.col("web_name"),
        ),
        on=["season", "element"],
        how="left",
    )
    for column in ["team", "opponent_team"]:
        df = df.join(
            static_teams.select(
                pl.col("season"),
                pl.col("id").alias(column),
                pl.col("name").alias(f"{column}_name"),
            ),
            on=["season", column],
            how="left",
        )

    # Save player predictions as a CSV file
    player_predictions = df.select(
        "season",
        "gameweek",
        "web_name",
        "team_name",
        "opponent_team_name",
        "predicted_total_points",
        "predicted_0_minutes",
        "predicted_1_to_59_minutes",
        "predicted_60_plus_minutes",
        "predicted_goals_scored",
        "predicted_assists",
        "predicted_clean_sheets",
        "predicted_goals_conceded",
        "predicted_saves",
        "predicted_bps",
        "predicted_bonus",
        "predicted_defensive_contribution",
        "predicted_defensive_contribution_threshold_prob",
    )
    player_predictions = player_predictions.sort(
        ["season", "gameweek", "predicted_total_points"],
        descending=[False, False, True],
    )
    path = OUTPUT_DIR / "player_predictions.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    player_predictions.write_csv(path, float_precision=2)

    # Save match predictions as a CSV file
    team_predictions = df.group_by(["season", "gameweek", "fixture"]).first()
    team_predictions = team_predictions.select(
        "season",
        "gameweek",
        "team_name",
        "opponent_team_name",
        "predicted_team_goals_scored",
        "predicted_opponent_goals_scored",
        "predicted_team_clean_sheets",
        "predicted_opponent_clean_sheets",
    )
    team_predictions = team_predictions.sort(
        ["season", "gameweek", "team_name"],
        descending=[False, False, False],
    )
    path = OUTPUT_DIR / "team_predictions.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    team_predictions.write_csv(path, float_precision=2)
