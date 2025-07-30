import polars as pl


def compute_imputed_set_piece_order(
    df: pl.LazyFrame,
) -> pl.LazyFrame:
    """Fill in missing set piece orders."""
    # Sort by kickoff time for correct forward filling
    df = df.sort("kickoff_time")

    for column in [
        "penalties_order",
        "direct_freekicks_order",
        "corners_and_indirect_freekicks_order",
    ]:
        # Fill in set piece orders for upcoming gameweeks
        df = df.with_columns(
            pl.col(column)
            .forward_fill()
            .over(["season", "element"])
            .alias(f"imputed_{column}"),
        )
        # Create a new column to indicate missing values
        df = df.with_columns(
            pl.col(f"imputed_{column}")
            .is_null()
            .cast(pl.Int8)
            .alias(f"{column}_missing"),
        )
        # Fill in remaining missing values with a default value of 11
        df = df.with_columns(
            pl.col(f"imputed_{column}").fill_null(11),
        )

    return df
