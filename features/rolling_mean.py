import polars as pl


def rolling_mean(
    df: pl.LazyFrame,
    order_by: str,
    group_by: str,
    columns: list[str],
    windows: list[int],
    defaults: list[int],
) -> pl.LazyFrame:
    """Calculate rolling means for specified columns and windows."""

    # Sort rows by the specified order (e.g. kickoff_time)
    df = df.sort(order_by)

    # Compute rolling averages for each column
    expressions = [
        pl.col(column)
        .rolling_mean(window, min_samples=1)
        # Important: To avoid data leakage, shift the rolling mean by 1
        .shift(1, fill_value=default)
        .over(group_by)
        .alias(f"{column}_rolling_mean_{window}")
        for column, window, default in zip(columns, windows, defaults, strict=True)
    ]
    df = df.with_columns(expressions)

    return df
