import polars as pl
from sklearn.base import BaseEstimator


def make_predictions(features: pl.DataFrame, model: BaseEstimator) -> pl.DataFrame:
    """Predict player points for each gameweek."""
    # Predict the total points for each player in each fixture
    predictions = model.predict(features)
    # Create a DataFrame with the predictions
    df = features.select(["season", "round", "element"]).with_columns(
        pl.Series("predicted_total_points", predictions),
    )
    # Sum up the number of points for each player in each gameweek
    df = df.group_by(["season", "round", "element"]).agg(
        pl.col("predicted_total_points").sum(),
    )
    return df
