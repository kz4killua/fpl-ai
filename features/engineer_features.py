import polars as pl

from .rolling_mean import rolling_mean


def engineer_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """Engineer features for modeling."""
    return df.pipe(
        rolling_mean,
        order_by="kickoff_time",
        group_by="code",
        columns=["total_points", "total_points"],
        window_sizes=[5, 38],
        defaults=[0, 0],
    )
