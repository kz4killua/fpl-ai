import polars as pl


def compute_rolling_std(
    df: pl.LazyFrame,
    columns: list[str],
    windows: list[int],
    over: list[str] = ("season", "code"),
    condition: pl.Expr | None = None,
    suffix: str = "",
) -> pl.LazyFrame:
    """Calculate rolling standard deviations ignoring null values."""
    if condition is None:
        condition = pl.lit(True)

    df = df.sort("kickoff_time").with_row_index("index")

    # Filter out invalid rows. We only do this once for efficiency.
    has_valid_data = pl.any_horizontal([pl.col(c).is_not_null() for c in set(columns)])
    subset = df.filter(condition & has_valid_data)

    # Create expressions for computing rolling standard deviations
    rolling_std_expressions = []
    resulting_columns = []

    for c, w in zip(columns, windows, strict=True):
        alias = f"{c}_rolling_std_{w}{suffix}"
        rolling_std_expressions.append(
            pl.col(c).rolling_std(window_size=w, min_periods=1).over(over).alias(alias)
        )
        resulting_columns.append(alias)

    # Compute all rolling stds in one go
    subset = subset.with_columns(rolling_std_expressions)

    # Join back to the original dataframe
    df = df.join(subset.select(["index"] + resulting_columns), on="index", how="left")

    # Shift and forward fill to avoid data leakage
    fill_expressions = []
    for c in resulting_columns:
        fill_expressions.append(pl.col(c).shift(1).forward_fill().over(over))

    df = df.with_columns(fill_expressions).drop("index")

    return df