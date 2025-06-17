import polars as pl


def compute_rolling_mean(
    df: pl.LazyFrame,
    columns: list[str],
    window_sizes: list[int],
) -> pl.LazyFrame:
    """Calculate rolling means ignoring null values."""

    # Sort rows for forward-filling
    df = df.sort("kickoff_time")

    # Add a temporary index column (to skip null values)
    df = df.with_row_index("index")

    for column, window_size in zip(columns, window_sizes, strict=True):
        alias = f"{column}_rolling_mean_{window_size}"
        # Compute the rolling mean for non-null values
        non_null = df.select(["index", "season", "code", column]).filter(
            pl.col(column).is_not_null()
        )
        non_null = non_null.with_columns(
            pl.col(column)
            .rolling_mean(window_size, min_samples=1)
            .over(["season", "code"])
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
            .shift(1)
            .forward_fill()
            .over(["season", "code"])
        )

    # Drop the temporary index column
    df = df.drop("index")

    return df
