import polars as pl


def compute_last_season_mean(
    df: pl.LazyFrame,
    columns: list[str],
    condition: pl.Expr | None = None,
    suffix: str = "",
):
    """Compute mean stats over the each player's previous season."""
    if condition is None:
        condition = pl.lit(True)
    # Compute player means for each season
    mapping = (
        df.filter(condition)
        .select(["season", "code", *columns])
        .group_by(["season", "code"])
        .agg([pl.col(c).mean().alias(f"{c}_mean_last_season{suffix}") for c in columns])
    )
    # Increment the season column
    mapping = mapping.with_columns(pl.col("season") + 1)
    # Map values to the original frame
    df = df.join(mapping, on=["season", "code"], how="left")
    return df
