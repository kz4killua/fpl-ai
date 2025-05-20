import polars as pl

from .rolling_mean import rolling_mean


def engineer_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """Engineer features for modeling."""
    return df.pipe(
        rolling_mean,
        order_by="kickoff_time",
        group_by="code",
        columns=[
            "total_points",
            "minutes",
            "uds_xG",
            "uds_xA",
            "clean_sheets",
            "goals_conceded",
            "saves",
            "bonus",
            "influence",
            "creativity",
            "threat",
            "ict_index",
        ],
        window_sizes=[5, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        defaults=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    )
