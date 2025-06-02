import polars as pl


def compute_rolling_mean(
    df: pl.LazyFrame,
    order_by: str,
    group_by: str,
    columns: list[str],
    window_sizes: list[int],
    defaults: list[int],
) -> pl.LazyFrame:
    """Calculate rolling means for specified columns and windows."""

    # Sort rows by the specified order (e.g. kickoff_time)
    df = df.sort(order_by)

    # Compute rolling averages for each column
    expressions = [
        pl.col(column)
        .rolling_mean(window_size, min_samples=1)
        # Important: To avoid data leakage, shift the rolling mean by 1
        .shift(1, fill_value=default)
        .forward_fill()
        .over(group_by)
        .alias(f"{column}_rolling_mean_{window_size}")
        for column, window_size, default in zip(
            columns, window_sizes, defaults, strict=True
        )
    ]
    df = df.with_columns(expressions)

    return df
