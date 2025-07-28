from typing import Any

import polars as pl


def compute_balanced_mean(
    df: pl.LazyFrame,
    this_season_column: str,
    last_season_column: str,
    decay: float,
    default: Any | None,
) -> pl.LazyFrame:
    """Balance values (e.g. averages) between seasons."""

    # Compute weights for the current and previous season
    last_season_weight = decay ** pl.col("record_count")
    this_season_weight = 1 - last_season_weight

    # Weight the values for the previous and current season
    last_season_value = last_season_weight * pl.col(last_season_column)
    this_season_value = this_season_weight * pl.col(this_season_column)

    # Fill missing values if a default is provided
    if default is not None:
        last_season_value = last_season_value.fill_null(default)
        this_season_value = this_season_value.fill_null(default)

    # Create a new column with the balanced result
    balanced_mean = last_season_value + this_season_value
    balanced_mean = balanced_mean.alias(f"balanced_{this_season_column}")
    df = df.with_columns(balanced_mean)

    return df
