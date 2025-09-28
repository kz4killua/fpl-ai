import polars as pl


def compute_fatigue(df: pl.DataFrame, window: int):
    """Compute the total number of minutes played in the last N days."""

    # Sort by kickoff time if necessary
    if not df.get_column("kickoff_time").is_sorted():
        df = df.sort("kickoff_time")

    # Compute rolling minutes over the given number of days
    alias = f"minutes_sum_{window}_days"
    fatigue = df.rolling(
        index_column="kickoff_time",
        period=f"{window}d",
        closed="left",
        group_by=("season", "code"),
    ).agg(pl.sum("minutes").alias(alias))

    # Note: A join ensures that the computed values align with the original DataFrame
    return df.join(
        fatigue.select("season", "code", "kickoff_time", alias),
        on=("season", "code", "kickoff_time"),
        how="left",
    )
