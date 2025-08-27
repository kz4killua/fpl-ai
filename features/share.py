import polars as pl


def compute_share(df: pl.LazyFrame, columns: list[str]):
    """Compute each player's percentage share to the team per-fixture."""
    # Aggregate team totals in each match
    totals = df.group_by(["season", "team_code", "fixture"]).agg(
        pl.col(column).sum() for column in columns
    )
    # Add team totals to the players frame
    df = df.join(
        totals.select(
            [
                pl.col("season"),
                pl.col("team_code"),
                pl.col("fixture"),
                *[pl.col(column).alias(f"team_{column}") for column in columns],
            ]
        ),
        on=["season", "team_code", "fixture"],
        how="left",
    )
    # Compute player share ratios
    df = df.with_columns(
        [
            (pl.col(column) / pl.col(f"team_{column}"))
            .fill_nan(0)
            .alias(f"{column}_share")
            for column in columns
        ]
    )
    # Drop unnecessary columns
    df = df.drop([pl.col(f"team_{column}") for column in columns])
    return df
