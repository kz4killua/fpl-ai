from collections.abc import Sequence

import polars as pl

from features.rolling import _compute_rolling_stat_with_nulls_ignored


def compute_rolling_std(
    df: pl.LazyFrame,
    columns: Sequence[str],
    window_sizes: Sequence[int],
    over: Sequence[str] = ("season", "code"),
    condition: pl.Expr | None = None,
    suffix: str = "",
) -> pl.LazyFrame:
    """Calculate rolling deviations ignoring null values."""
    return _compute_rolling_stat_with_nulls_ignored(
        df,
        columns,
        window_sizes,
        stat="std",
        over=over,
        condition=condition,
        suffix=suffix,
    )
