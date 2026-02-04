from collections.abc import Sequence

import polars as pl

from features.rolling import _compute_rolling_stat_with_nulls_ignored


def compute_rolling_mean(
    df: pl.LazyFrame,
    columns: Sequence[str],
    window_sizes: Sequence[int],
    over: Sequence[str] = ("season", "code"),
    condition: pl.Expr | None = None,
    suffix: str = "",
) -> pl.LazyFrame:
    """Calculate rolling means ignoring null values."""
    return _compute_rolling_stat_with_nulls_ignored(
        df,
        columns,
        window_sizes,
        stat="mean",
        over=over,
        condition=condition,
        suffix=suffix,
    )
