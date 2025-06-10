import polars as pl


def compute_rolling_mean(
    df: pl.LazyFrame,
    order_by: str,
    group_by: str,
    columns: list[str],
    window_sizes: list[int],
    defaults: list[int],
) -> pl.LazyFrame:
    """Calculate rolling means ignoring null values."""

    # Sort rows by the specified column (e.g. kickoff_time)
    df = df.sort(order_by)

    # Add a temporary index column (to skip null values)
    df = df.with_row_index("index")

    for column, window_size, default in zip(
        columns, window_sizes, defaults, strict=True
    ):
        alias = f"{column}_rolling_mean_{window_size}"
        # Compute the rolling mean for non-null values
        non_null = df.select(["index", column, group_by, order_by]).filter(
            pl.col(column).is_not_null()
        )
        non_null = non_null.with_columns(
            pl.col(column)
            .rolling_mean(window_size, min_samples=1)
            .over(group_by)
            .alias(alias)
        )
        # Add the non-null results to the original frame
        df = df.join(
            non_null.select(["index", alias]),
            on="index",
            how="left",
        )
        df = df.with_columns(
            pl.col(alias)
            # Important: To avoid data leakage, shift the rolling mean by 1
            .shift(1, fill_value=default)
            .forward_fill()
            .over(group_by)
        )

    # Drop the temporary index column
    df = df.drop("index")

    return df
