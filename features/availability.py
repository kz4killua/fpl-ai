import polars as pl

EXPECTED_BACK_PATTERN = r"Expected back (\d{1,2}) ([A-Za-z]{3})"
SUSPENDED_UNTIL_PATTERN = r"Suspended until (\d{1,2}) ([A-Za-z]{3})"
MONTH_MAPPING = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}


def compute_availability(df: pl.LazyFrame) -> pl.LazyFrame:
    """Compute the availability of players for each fixture."""

    # Sort by kickoff time (for forward filling)
    df = df.sort("kickoff_time")

    # Forward fill the status, news, and news_added columns for upcoming fixtures
    df = df.with_columns(
        [
            pl.col("status").forward_fill().over("code"),
            pl.col("news").forward_fill().over("code"),
            pl.col("news_added").forward_fill().over("code"),
        ]
    )

    # Parse the "news" column to get expected return days and months
    df = df.with_columns(
        [
            pl.col("news")
            .str.extract(EXPECTED_BACK_PATTERN, 1)
            .cast(pl.Int32)
            .alias("expected_back_day"),
            pl.col("news")
            .str.extract(EXPECTED_BACK_PATTERN, 2)
            .replace(MONTH_MAPPING)
            .alias("expected_back_month"),
            pl.col("news")
            .str.extract(SUSPENDED_UNTIL_PATTERN, 1)
            .cast(pl.Int32)
            .alias("suspended_until_day"),
            pl.col("news")
            .str.extract(SUSPENDED_UNTIL_PATTERN, 2)
            .replace(MONTH_MAPPING)
            .alias("suspended_until_month"),
        ]
    )
    df = df.with_columns(
        [
            pl.when(
                pl.col("suspended_until_day").is_not_null()
                & pl.col("suspended_until_month").is_not_null()
            )
            .then(pl.col("suspended_until_day"))
            .when(
                pl.col("expected_back_day").is_not_null()
                & pl.col("expected_back_month").is_not_null()
            )
            .then(pl.col("expected_back_day"))
            .otherwise(None)
            .alias("return_day"),
            pl.when(
                pl.col("suspended_until_day").is_not_null()
                & pl.col("suspended_until_month").is_not_null()
            )
            .then(pl.col("suspended_until_month"))
            .when(
                pl.col("expected_back_day").is_not_null()
                & pl.col("expected_back_month").is_not_null()
            )
            .then(pl.col("expected_back_month"))
            .otherwise(None)
            .alias("return_month"),
        ]
    )

    # Calculate the return dates
    df = df.with_columns(
        pl.when(pl.col("news_added").is_not_null())
        .then(pl.col("news_added").dt.year())
        .otherwise(pl.lit(None))
        .alias("news_added_year"),
    )
    df = df.with_columns(
        pl.datetime(
            pl.col("news_added_year"),
            pl.col("return_month"),
            pl.col("return_day"),
        ).alias("return_date"),
    )

    # Drop time zones to avoid issues with comparisons
    df = df.with_columns(
        [
            pl.col("news_added").dt.replace_time_zone(None),
            pl.col("return_date").dt.replace_time_zone(None),
            pl.col("kickoff_time").dt.replace_time_zone(None),
        ]
    )

    # Roll over the return dates when necessary
    df = df.with_columns(
        pl.when(
            pl.col("return_date").is_not_null()
            & (pl.col("return_date") < pl.col("news_added"))
        )
        .then(pl.col("return_date").dt.offset_by("1y"))
        .otherwise(pl.col("return_date"))
        .alias("return_date"),
    )

    # Compute player availability for each fixture
    df = df.with_columns(
        pl.when(
            pl.col("return_date").is_not_null()
            & (pl.col("return_date") <= pl.col("kickoff_time"))
        )
        .then(pl.lit(100))
        .when(pl.col("status") == "a")
        .then(pl.lit(100))
        .when(pl.col("status") == "u")
        .then(pl.lit(0))
        .when(pl.col("status") == "d")
        .then(pl.col("chance_of_playing_next_round").fill_null(100))
        .when(pl.col("status").is_null())
        .then(pl.lit(None))
        .otherwise(pl.lit(0))
        .alias("availability")
    )

    # Drop the columns we no longer need
    df = df.drop(
        [
            "news_added_year",
            "expected_back_day",
            "expected_back_month",
            "suspended_until_day",
            "suspended_until_month",
            "return_day",
            "return_month",
            "return_date",
        ]
    )

    return df
