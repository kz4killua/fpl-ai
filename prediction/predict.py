import polars as pl

from prediction.model import PredictionModel


def make_predictions(
    model: PredictionModel, players: pl.DataFrame, matches: pl.DataFrame
) -> pl.DataFrame:
    """Predict player points for each gameweek."""
    # Predict total points for each player in each fixture
    predictions = model.predict(players, matches)
    df = players.select(["season", "round", "element"]).with_columns(
        predictions.alias("total_points")
    )
    # Sum up the number of points for each player in each gameweek
    df = df.group_by(["season", "round", "element"]).agg(
        pl.col("total_points").sum(),
    )
    return df
