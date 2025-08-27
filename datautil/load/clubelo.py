from datetime import datetime

import polars as pl

from datautil.constants import DATA_DIR


def load_clubelo(cutoff_time: datetime) -> pl.LazyFrame:
    """Load local clubelo ratings for teams."""
    ratings = pl.scan_csv(
        DATA_DIR / "clubelo/ratings/*.csv",
        try_parse_dates=True,
        raise_if_empty=False,
        null_values=["None"],
    )
    # Add FPL team codes to ratings
    team_ids = pl.scan_csv(
        DATA_DIR / "clubelo/team_ids.csv",
        try_parse_dates=True,
        raise_if_empty=False,
    )
    ratings = ratings.join(
        team_ids.select(
            pl.col("clubelo_name").alias("Club"),
            pl.col("fpl_code"),
        ),
        how="left",
        on="Club",
    )
    # Filter ratings using the cutoff time
    ratings = pl.concat(
        [
            ratings.filter(pl.col("To") < cutoff_time.date()),
            ratings.filter(pl.col("To") >= cutoff_time.date())
            .group_by("Club")
            .agg(pl.min("To"))
            .join(ratings, on=["Club", "To"])
        ],
        how="diagonal"
    )
    return ratings
