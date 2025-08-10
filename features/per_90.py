import polars as pl


def compute_per_90(df: pl.LazyFrame, columns: list[str]) -> pl.LazyFrame:
    """Compute per 90 statistics for players."""
    df = df.with_columns(
        [
            pl.when(pl.col("minutes").is_null())
            .then(None)
            .when(pl.col("minutes") == 0)
            .then(None)
            .otherwise(pl.col(column) / pl.col("minutes") * 90)
            .alias(f"{column}_per_90")
            for column in columns
        ]
    )
    return df
