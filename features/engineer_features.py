import polars as pl

from .rolling_mean import rolling_mean


def engineer_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """Engineer features for modeling."""
    return df.pipe(
        rolling_mean,
        order_by="kickoff_time",
        group_by="element",
        columns=["total_points"],
        windows=[4],
        defaults=[0],
    )
