import polars as pl


def compute_record_count(df: pl.LazyFrame):
    """Count the number of records for each player."""
    df = df.sort("kickoff_time")
    return df.with_columns(
        (
            pl.col("total_points").is_not_null().cum_sum().over(["season", "code"])
            - pl.col("total_points").is_not_null()
        ).alias("record_count")
    )
