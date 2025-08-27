import polars as pl


def compute_one_hot_minutes_category(df: pl.DataFrame):
    """Compute one-hot encoded categories for minute categories."""
    values = [
        "0_minutes",
        "1_to_59_minutes",
        "60_plus_minutes",
    ]
    for value in values:
        df = df.with_columns(
            pl.when(pl.col("minutes_category").is_null())
            .then(pl.lit(None))
            .when(pl.col("minutes_category") == value)
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias(f"minutes_category_{value}"),
        )
    return df
