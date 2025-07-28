import polars as pl


def compute_imputed_set_piece_order(
    df: pl.LazyFrame,
) -> pl.LazyFrame:
    """Fill in missing set piece orders."""
    for column in [
        "penalties_order",
        "direct_freekicks_order",
        "corners_and_indirect_freekicks_order",
    ]:
        # Create a missing indicator column
        df = df.with_columns(
            pl.col(column).is_null().cast(pl.Int8).alias(f"{column}_missing")
        )
        # Fill missing values with a default value of 11
        df = df.with_columns(pl.col(column).fill_null(11).alias(f"imputed_{column}"))
    return df
