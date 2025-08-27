import glob

import polars as pl

from datautil.constants import DATA_DIR
from datautil.utils import convert_season_to_year, convert_year_col_to_season_col


def load_understat(seasons: list[str]):
    """Loads local understat data for the given seasons."""

    # Load local data
    players = load_players(seasons)
    teams = load_teams(seasons)
    fixtures = load_fixtures(seasons)
    player_ids = load_player_ids()
    team_ids = load_team_ids()
    fixture_ids = load_fixture_ids(seasons)

    # Convert understat seasons to FPL seasons
    players = players.with_columns(
        convert_year_col_to_season_col("season").alias("fpl_season")
    )
    teams = teams.with_columns(
        convert_year_col_to_season_col("season").alias("fpl_season")
    )
    fixtures = fixtures.with_columns(
        convert_year_col_to_season_col("season").alias("fpl_season")
    )
    fixture_ids = fixture_ids.with_columns(
        convert_year_col_to_season_col("season").alias("fpl_season")
    )

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
                pl.col("fpl_season"),
                pl.col("understat_id").alias("fixture_id"),
                pl.col("fpl_id").alias("fpl_fixture_id"),
            ]
        ),
        how="left",
        on=["fpl_season", "fixture_id"],
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
    teams = teams.join(
        fixtures.select(
            [
                pl.col("h").alias("id"),
                pl.col("datetime").alias("date"),
                pl.col("id").alias("fixture_id_h"),
            ]
        ),
        how="left",
        on=["id", "date"],
    )
    teams = teams.join(
        fixtures.select(
            [
                pl.col("a").alias("id"),
                pl.col("datetime").alias("date"),
                pl.col("id").alias("fixture_id_a"),
            ]
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
                pl.col("fpl_season"),
                pl.col("understat_id").alias("fixture_id"),
                pl.col("fpl_id").alias("fpl_fixture_id"),
            ]
        ),
        how="left",
        on=["fpl_season", "fixture_id"],
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

    return players, teams


def load_players(seasons: list[str]) -> pl.LazyFrame:
    """Load player match data."""
    seasons = list(map(convert_season_to_year, seasons))
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


def load_teams(seasons: list[str]) -> pl.LazyFrame:
    """Load team data."""
    seasons = list(map(convert_season_to_year, seasons))

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
    seasons = list(map(convert_season_to_year, seasons))

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
    """Load FPL-to-understat player ID mappings."""
    return pl.scan_csv(DATA_DIR / "understat/player_ids.csv")


def load_team_ids() -> pl.LazyFrame:
    """Load FPL-to-understat team ID mappings."""
    return pl.scan_csv(DATA_DIR / "understat/team_ids.csv")


def load_fixture_ids(seasons: list[str]) -> pl.LazyFrame:
    """Load FPL-to-understat fixture ID mappings."""
    seasons = list(map(convert_season_to_year, seasons))

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
