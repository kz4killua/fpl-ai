from collections.abc import Sequence
from typing import Literal

import polars as pl

from loaders.utils import force_dataframe


def _compute_rolling_stat_with_nulls_ignored(
    df: pl.LazyFrame,
    columns: Sequence[str],
    window_sizes: Sequence[int],
    stat: Literal["mean", "std"],
    over: Sequence[str] = ("season", "code"),
    condition: pl.Expr | None = None,
    suffix: str = "",
) -> pl.LazyFrame:
    """Calculate rolling means or stds ignoring null values."""
    df = force_dataframe(df)

    # Sort rows for forward-filling
    df = df.sort("kickoff_time")

    # Add a temporary index column (to skip null values)
    df = df.with_row_index("index")

    if condition is None:
        condition = pl.lit(True)

    results = []
    for column, window_size in zip(columns, window_sizes, strict=True):
        alias = _alias(column, window_size, stat, suffix)

        # Filter for non-null values meeting the condition (if any)
        subset = df.filter(pl.col(column).is_not_null() & condition)
        subset = subset.select(["index", *over, column])

        # Apply the specific rolling function
        col_expr = pl.col(column)
        if stat == "mean":
            rolled = col_expr.rolling_mean(window_size=window_size, min_samples=1)
        elif stat == "std":
            rolled = col_expr.rolling_std(window_size=window_size, min_samples=1)
        else:
            raise ValueError(f"Unsupported stat: {stat}")

        subset = subset.with_columns(rolled.over(over).alias(alias))
        results.append(subset.select(["index", alias]))

    # Join all results back to the original frame
    for result in results:
        df = df.join(result, on="index", how="left")

    # Shift and forward-fill to avoid data leakage
    expressions = []
    for column, window_size in zip(columns, window_sizes, strict=True):
        alias = _alias(column, window_size, stat, suffix)
        expressions.append(pl.col(alias).shift(1).forward_fill().over(over))

    df = df.with_columns(expressions)

    # Drop the temporary index column
    df = df.drop("index")

    return df


def _alias(column: str, window_size: int, stat: str, suffix: str) -> str:
    return f"{column}_rolling_{stat}_{window_size}{suffix}"
