from typing import Any

import polars as pl


def compute_balanced_rolling_mean(
    df: pl.LazyFrame,
    columns: list[str],
    defaults: list[Any],
    window_sizes: list[int],
    decay: float,
) -> pl.LazyFrame:
    """Balance rolling means with the last season's averages."""

    # Compute weights for the current and previous season
    last_season_weight = decay ** pl.col("record_count")
    this_season_weight = 1 - last_season_weight

    for column, window_size, default in zip(
        columns, window_sizes, defaults, strict=True
    ):
        # Weight the values for the previous and current season
        last_season_value = last_season_weight * pl.col(
            f"imputed_{column}_mean_last_season"
        )
        this_season_value = this_season_weight * pl.col(
            f"{column}_rolling_mean_{window_size}"
        )
        # Fill missing values if a default is provided
        if default is not None:
            last_season_value = last_season_value.fill_null(default)
            this_season_value = this_season_value.fill_null(default)
        # Create a new column with the balanced result
        balanced_rolling_mean = last_season_value + this_season_value
        balanced_rolling_mean = balanced_rolling_mean.alias(
            f"{column}_balanced_rolling_mean_{window_size}"
        )
        df = df.with_columns(balanced_rolling_mean)

    return df
