import polars as pl


def compute_depth_rank(df: pl.LazyFrame, column: str) -> pl.LazyFrame:
    """Rank players based on metrics within teams and positions."""

    group = ["season", "fixture", "team", "element_type"]
    available = pl.col("availability") == 100

    # Rank players within each team and position.
    df = df.with_columns(
        pl.col(column)
        .rank(method="dense", descending=True)
        .over(group)
        .alias(f"depth_rank_{column}")
    )

    # Calculated an adjusted rank, only considering available players.
    df = df.with_columns(
        (pl.when(available).then(pl.col(column)).otherwise(pl.lit(None)))
        .rank(method="dense", descending=True)
        .over(group)
        .alias(f"adjusted_depth_rank_{column}")
    )

    # Calculate the rank change after adjustments.
    df = df.with_columns(
        (
            pl.col(f"depth_rank_{column}") - pl.col(f"adjusted_depth_rank_{column}")
        ).alias(f"depth_rank_change_{column}")
    )

    return df
