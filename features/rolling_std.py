import polars as pl

from loaders.utils import force_dataframe


def compute_rolling_std(
    df: pl.LazyFrame,
    columns: list[str],
    window_sizes: list[int],
    over: list[str] = ("season", "code"),
    condition: pl.Expr | None = None,
    suffix: str = "",
) -> pl.LazyFrame:
    """Calculate rolling stds ignoring null values."""
    df = force_dataframe(df)

    # Sort rows for forward-filling
    df = df.sort("kickoff_time")

    # Add a temporary index column (to skip null values)
    df = df.with_row_index("index")

    if condition is None:
        condition = pl.lit(True)

    results = []
    for column, window_size in zip(columns, window_sizes, strict=True):
        alias = _alias(column, window_size, suffix)
        # Compute the rolling deviations for selected values
        subset = df.filter(pl.col(column).is_not_null() & condition)
        subset = subset.select(["index", *over, column])
        subset = subset.with_columns(
            pl.col(column)
            .rolling_std(window_size, min_samples=1)
            .over(over)
            .alias(alias)
        )
        results.append(subset.select(["index", alias]))

    # Join all results back to the original frame
    for result in results:
        df = df.join(result, on="index", how="left")

    # Shift and forward-fill to avoid data leakage
    expressions = []
    for column, window_size in zip(columns, window_sizes, strict=True):
        alias = _alias(column, window_size, suffix)
        expressions.append(pl.col(alias).shift(1).forward_fill().over(over))

    df = df.with_columns(expressions)

    # Drop the temporary index column
    df = df.drop("index")

    return df.lazy()


def _alias(column: str, window_size: int, suffix: str) -> str:
    return f"{column}_rolling_std_{window_size}{suffix}"
