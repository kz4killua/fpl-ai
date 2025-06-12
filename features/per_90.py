import polars as pl


def compute_per_90(df: pl.LazyFrame, columns: list[str]) -> pl.LazyFrame:
    """Compute per 90 statistics for players."""
    df = df.with_columns(
        [
            pl.when(pl.col("minutes") > 0)
            .then((pl.col(column) / pl.col("minutes") * 90).alias(f"{column}_per_90"))
            .otherwise(0)
            for column in columns
        ]
    )
    return df
