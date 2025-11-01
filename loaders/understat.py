import glob
from datetime import datetime

import polars as pl

from loaders.constants import DATA_DIR


def load_understat(seasons: list[int], cutoff_time: datetime):
    """Loads local understat data for the given seasons."""

    # Load local data
    players = load_players(seasons)
    teams = load_teams(seasons)
    fixtures = load_fixtures(seasons)
    player_ids = load_player_ids()
    team_ids = load_team_ids()
    fixture_ids = load_fixture_ids(seasons)

    # Add FPL player codes to players
    players = players.join(
        player_ids.select([pl.col("understat_id").alias("id"), pl.col("fpl_code")]),
        how="left",
        on="id",
    )

    # Add FPL fixture IDs to players
    players = players.join(
        fixture_ids.select(
            [
                pl.col("season"),
                pl.col("understat_id").alias("fixture_id"),
                pl.col("fpl_id").alias("fpl_fixture_id"),
            ]
        ),
        how="left",
        on=["season", "fixture_id"],
    )

    # Any unmapped fixtures are for matches outside the EPL. Remove them.
    players = players.filter(pl.col("fpl_fixture_id").is_not_null())

    # Extract PPDA stats and add to teams
    pattern = r"\{'att':\s*(\d+),\s*'def':\s*(\d+)\}"
    teams = teams.with_columns(
        pl.col("ppda").str.extract(pattern, 1).cast(pl.Int32).alias("ppda_att"),
        pl.col("ppda").str.extract(pattern, 2).cast(pl.Int32).alias("ppda_def"),
        pl.col("ppda_allowed")
        .str.extract(pattern, 1)
        .cast(pl.Int32)
        .alias("ppda_allowed_att"),
        pl.col("ppda_allowed")
        .str.extract(pattern, 2)
        .cast(pl.Int32)
        .alias("ppda_allowed_def"),
    ).drop(["ppda", "ppda_allowed"])

    # Add understat fixture IDs to teams
    for column in ["h", "a"]:
        teams = teams.join(
            fixtures.select(
                pl.col(column).alias("id"),
                pl.col("datetime").alias("date"),
                pl.col("id").alias(f"fixture_id_{column}"),
            ),
            how="left",
            on=["id", "date"],
        )

    teams = teams.with_columns(
        pl.when(pl.col("h_a") == "h")
        .then(pl.col("fixture_id_h"))
        .otherwise(pl.col("fixture_id_a"))
        .alias("fixture_id"),
    )
    teams = teams.drop(["fixture_id_h", "fixture_id_a"])

    # Add FPL fixture IDs to teams
    teams = teams.join(
        fixture_ids.select(
            [
                pl.col("season"),
                pl.col("understat_id").alias("fixture_id"),
                pl.col("fpl_id").alias("fpl_fixture_id"),
            ]
        ),
        how="left",
        on=["season", "fixture_id"],
    )

    # Add FPL team codes to teams
    teams = teams.join(
        team_ids.select(
            [
                pl.col("understat_id").alias("id"),
                pl.col("fpl_code"),
            ]
        ),
        how="left",
        on="id",
    )

    # Filter records using the cutoff time
    # TODO: Push this into the respective functions
    players = players.filter(pl.col("date") < cutoff_time.date())
    teams = teams.filter(pl.col("date") < cutoff_time.date())

    return players, teams


def load_players(seasons: list[int]) -> pl.LazyFrame:
    """Load player data."""
    return (
        pl.scan_csv(
            DATA_DIR / "understat/player/matches/*.csv",
            try_parse_dates=True,
            raise_if_empty=False,
            include_file_paths="file_path",
        )
        .with_columns(
            # Rename the fixture ID column to avoid confusion
            pl.col("id").alias("fixture_id"),
            # Extract player ID from file path
            pl.col("file_path").str.extract(r"(\d+)\.csv").cast(pl.Int32).alias("id"),
        )
        .drop("file_path")
        .filter(pl.col("season").is_in(seasons))
    )


def load_teams(seasons: list[int]) -> pl.LazyFrame:
    """Load team data."""

    frames = []
    for season in seasons:
        path = DATA_DIR / f"understat/season/{season}/teams/*.csv"
        if glob.glob(str(path)):
            frames.append(
                pl.scan_csv(
                    path,
                    try_parse_dates=True,
                    raise_if_empty=False,
                ).with_columns(
                    pl.lit(season).alias("season"),
                )
            )

    return pl.concat(frames, how="diagonal")


def load_fixtures(seasons: list[str]) -> pl.LazyFrame:
    """Load fixture data."""

    frames = []
    for season in seasons:
        path = DATA_DIR / f"understat/season/{season}/dates.csv"
        if path.exists():
            frames.append(
                pl.scan_csv(
                    path,
                    try_parse_dates=True,
                    raise_if_empty=False,
                ).with_columns(
                    pl.lit(season).alias("season"),
                )
            )

    return pl.concat(frames, how="diagonal")


def load_player_ids() -> pl.LazyFrame:
    """Load FPL player ID mappings."""
    return pl.scan_csv(DATA_DIR / "understat/player_ids.csv")


def load_team_ids() -> pl.LazyFrame:
    """Load FPL team ID mappings."""
    return pl.scan_csv(DATA_DIR / "understat/team_ids.csv")


def load_fixture_ids(seasons: list[str]) -> pl.LazyFrame:
    """Load FPL fixture ID mappings."""

    frames = []
    for season in seasons:
        path = DATA_DIR / f"understat/season/{season}/fixture_ids.csv"
        if path.exists():
            frames.append(
                pl.scan_csv(
                    path,
                    try_parse_dates=True,
                    raise_if_empty=False,
                ).with_columns(
                    pl.lit(season).alias("season"),
                )
            )

    return pl.concat(frames, how="diagonal")
