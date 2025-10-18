import polars as pl


def compute_per_90(
    df: pl.LazyFrame, columns: list[str], threshold: int = 30
) -> pl.LazyFrame:
    """Compute per-90 statistics for players with minutes above the threshold."""
    df = df.with_columns(
        [
            pl.when(pl.col("minutes").is_null())
            .then(None)
            .when(pl.col("minutes") == 0)
            .then(None)
            .when(pl.col("minutes") < threshold)
            .then(None)
            .otherwise(pl.col(column) / pl.col("minutes") * 90)
            .alias(f"{column}_per_90")
            for column in columns
        ]
    )
    return df
