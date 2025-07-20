import polars as pl


def compute_minutes_category(df: pl.LazyFrame):
    """Categorize players based on their minutes played."""
    return df.with_columns(
        pl.when(pl.col("minutes") == 0)
        .then(pl.lit("0_minutes"))
        .when(pl.col("minutes").is_between(1, 59))
        .then(pl.lit("1_to_59_minutes"))
        .otherwise(pl.lit("60_plus_minutes"))
        .alias("minutes_category")
    )
