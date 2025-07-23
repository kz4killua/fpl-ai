import polars as pl


def compute_rolling_mean(
    df: pl.LazyFrame,
    columns: list[str],
    window_sizes: list[int],
    over: list[str] = ("season", "code"),
    condition: pl.Expr | None = None,
    suffix: str = "",
) -> pl.LazyFrame:
    """Calculate rolling means ignoring null values."""

    # Sort rows for forward-filling
    df = df.sort("kickoff_time")

    # Add a temporary index column (to skip null values)
    df = df.with_row_index("index")

    if condition is None:
        condition = pl.lit(True)

    for column, window_size in zip(columns, window_sizes, strict=True):
        alias = f"{column}_rolling_mean_{window_size}{suffix}"
        # Compute the rolling mean for selected values
        selected = df.filter(pl.col(column).is_not_null() & condition).select(
            ["index", *over, column]
        )
        selected = selected.with_columns(
            pl.col(column)
            .rolling_mean(window_size, min_samples=1)
            .over(over)
            .alias(alias)
        )
        # Add the selected results to the original frame
        df = df.join(
            selected.select(["index", alias]),
            on="index",
            how="left",
        )
        df = df.with_columns(
            pl.col(alias)
            # Important: To avoid data leakage, shift the rolling mean by 1
            .shift(1)
            .forward_fill()
            .over(over)
        )

    # Drop the temporary index column
    df = df.drop("index")

    return df
