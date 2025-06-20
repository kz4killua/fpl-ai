import polars as pl

from datautil.utils import (
    convert_season_col_to_year_col,
    convert_year_col_to_season_col,
)


def compute_last_season_mean(df: pl.LazyFrame, columns: list[str]):
    """Compute average stats over the each player's previous season."""
    # Compute player means for each season
    mapping = (
        df.select(["season", "code", *columns])
        .group_by(["season", "code"])
        .agg([pl.col(c).mean().alias(f"{c}_mean_last_season") for c in columns])
    )
    # Increment the season column
    mapping = (
        mapping.with_columns(convert_season_col_to_year_col("season").alias("year"))
        .with_columns((pl.col("year") + 1).alias("next_year"))
        .with_columns(convert_year_col_to_season_col("next_year").alias("next_season"))
        .with_columns(pl.col("next_season").alias("season"))
        .drop(["year", "next_year", "next_season"])
    )
    # Map values to the original frame
    df = df.join(mapping, on=["season", "code"], how="left")
    return df
