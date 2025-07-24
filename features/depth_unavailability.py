import polars as pl


def compute_depth_unavailability(df: pl.LazyFrame, column: str) -> pl.LazyFrame:
    """Sum up the unavailability of all players within teams and positions."""

    alias = f"depth_unavailability_{column}"
    group = ["season", "fixture", "team", "element_type"]
    unavailability = (100 - pl.col("availability")) / 100

    # Compute the total unavailability within each team for each fixture.
    mapping = (
        df.select([*group, column, "availability"])
        .group_by(group)
        .agg((pl.col(column) * (unavailability)).sum().alias(alias))
        .select([*group, alias])
    )

    # Map to players.
    df = df.join(mapping, on=group, how="left")
    return df
