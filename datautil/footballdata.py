from datetime import UTC, datetime, time

import polars as pl

from datautil.constants import DATA_DIR
from datautil.utils import calculate_implied_probabilities


def load_footballdata(seasons: list[int], cutoff_time: datetime) -> pl.LazyFrame:
    """Load historical odds and results from football-data.co.uk."""

    frames = []
    for season in seasons:
        path = DATA_DIR / f"footballdata/data/{season}.csv"
        frame = pl.scan_csv(path, try_parse_dates=True)
        frame = frame.with_columns(pl.lit(season).alias("season"))

        # For safe filtering, assume a midnight kickoff for rows without a Time value.
        if season < 2019:
            frame = frame.with_columns(pl.lit(time(0, 0, 0)).alias("Time"))

        frames.append(frame)

    df: pl.LazyFrame = pl.concat(frames, how="diagonal_relaxed")

    # Only include matches before the cutoff time
    df = df.with_columns(
        pl.col("Date").dt.combine(pl.col("Time")).alias("kickoff_time")
    )
    df = df.with_columns(pl.col("kickoff_time").dt.replace_time_zone("Europe/London"))
    df = df.with_columns(pl.col("kickoff_time").dt.convert_time_zone("UTC"))

    if cutoff_time.tzinfo != UTC:
        raise ValueError("cutoff_time must be in UTC")
    df = df.filter(pl.col("kickoff_time") <= cutoff_time)

    # Add FPL codes for home and away teams
    team_ids = pl.scan_csv(DATA_DIR / "footballdata/team_ids.csv")
    for column in ["HomeTeam", "AwayTeam"]:
        df = df.join(
            team_ids.select(
                pl.col("footballdata_name").alias(column),
                pl.col("fpl_code").alias(f"{column}_fpl_code"),
            ),
            on=column,
            how="left",
        )

    # Calculate implied probabilities for selected bookmakers
    bookmakers = [
        ("PSH", "PSD", "PSA"),
    ]
    for home, draw, away in bookmakers:
        implied_home, implied_away, implied_draw = calculate_implied_probabilities(
            pl.col(home),
            pl.col(away),
            pl.col(draw),
        )
        df = df.with_columns(
            implied_home.alias(f"{home}_implied"),
            implied_away.alias(f"{away}_implied"),
            implied_draw.alias(f"{draw}_implied"),
        )

    columns = [
        "season",
        "HomeTeam",
        "AwayTeam",
        "HomeTeam_fpl_code",
        "AwayTeam_fpl_code",
    ]
    for home, draw, away in bookmakers:
        columns.extend(
            [
                home,
                draw,
                away,
                f"{home}_implied",
                f"{draw}_implied",
                f"{away}_implied",
            ]
        )
    return df.select(columns)
