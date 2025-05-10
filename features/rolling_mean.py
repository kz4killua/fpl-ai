import polars as pl


def rolling_mean(
    df: pl.LazyFrame,
    order_by: str,
    group_by: str,
    columns: list[str],
    windows: list[int],
    defaults: list[int],
) -> pl.LazyFrame:
    """
    Calculate rolling means for specified columns and windows.

    Args:
        df (pl.LazyFrame): Input DataFrame.
        order_by (str): Column name to sort by before calculating rolling averages.
        group_by (str): Column name to group by for rolling averages.
        columns (list[str]): List of column names to calculate rolling averages for.
        windows (list[int]): List of window sizes for rolling averages.
        defaults (list[int]): List of default values for each column.

    Returns:
        pl.LazyFrame: DataFrame with rolling averages added.
    """

    # Sort the dataframe in a deterministic order
    df = df.sort(order_by)

    # Create a list of expressions for rolling averages
    expressions = [
        pl.col(column)
        .rolling_mean(window, min_samples=1)
        # Important: To avoid data leakage, shift the rolling mean by 1
        .shift(1, fill_value=default)
        .over(group_by)
        .alias(f"{column}_rolling_mean_{window}")
        for column, window, default in zip(columns, windows, defaults, strict=True)
    ]

    # Apply the rolling averages to the dataframe
    df = df.with_columns(expressions)

    return df
