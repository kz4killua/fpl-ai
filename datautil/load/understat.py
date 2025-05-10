import polars as pl

from datautil.constants import DATA_DIR
from datautil.utils import convert_season_to_year, convert_year_col_to_season_col


def load_understat(seasons: list[str]):
    """Loads local understat data for the given seasons."""

    # Load local data
    players = load_player_matches(seasons)
    teams = load_teams(seasons)
    dates = load_dates(seasons)
    player_ids = load_player_ids()
    fixture_ids = load_fixture_ids(seasons)
    team_ids = load_team_ids()

    # Add FPL seasons
    players = players.with_columns(
        convert_year_col_to_season_col("season").alias("fpl_season")
    )
    teams = teams.with_columns(
        convert_year_col_to_season_col("season").alias("fpl_season")
    )
    dates = dates.with_columns(
        convert_year_col_to_season_col("season").alias("fpl_season")
    )
    fixture_ids = fixture_ids.with_columns(
        convert_year_col_to_season_col("season").alias("fpl_season")
    )

    # Add FPL player codes to players
    players = players.join(
        player_ids.select(["understat_id", "fpl_code"]),
        how="left",
        left_on=["player_id"],
        right_on=["understat_id"],
    )

    # Add FPL fixture IDs to players
    players = players.join(
        fixture_ids.select(
            [
                pl.col("fpl_season"),
                pl.col("understat_id"),
                pl.col("fpl_id").alias("fpl_fixture_id"),
            ]
        ),
        how="left",
        left_on=["fpl_season", "id"],
        right_on=["fpl_season", "understat_id"],
    )

    # Any unmapped fixtures are for matches outside the EPL. Remove them.
    players = players.filter(pl.col("fpl_fixture_id").is_not_null())

    # Add PPDA stats to teams
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
        dates.select(
            [pl.col("id").alias("fixture_id_h"), pl.col("h"), pl.col("datetime")]
        ),
        how="left",
        left_on=["id", "date"],
        right_on=["h", "datetime"],
    )
    teams = teams.join(
        dates.select(
            [pl.col("id").alias("fixture_id_a"), pl.col("a"), pl.col("datetime")]
        ),
        how="left",
        left_on=["id", "date"],
        right_on=["a", "datetime"],
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
                pl.col("understat_id"),
                pl.col("fpl_id").alias("fpl_fixture_id"),
            ]
        ),
        how="left",
        left_on=["fpl_season", "fixture_id"],
        right_on=["fpl_season", "understat_id"],
    )

    # Add FPL team codes to teams
    teams = teams.join(
        team_ids.select(["understat_id", "fpl_code"]),
        how="left",
        left_on=["id"],
        right_on=["understat_id"],
    )

    return players, teams


def load_player_matches(seasons: list[str]) -> pl.LazyFrame:
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
            pl.col("file_path")
            .str.extract(r"(\d+)\.csv")
            .cast(pl.Int32)
            .alias("player_id"),
        )
        .filter(pl.col("season").is_in(seasons))
    )


def load_teams(seasons: list[str]) -> pl.LazyFrame:
    """Load team data."""
    seasons = list(map(convert_season_to_year, seasons))
    frames = [
        pl.scan_csv(
            DATA_DIR / f"understat/season/{season}/teams/*.csv",
            try_parse_dates=True,
            raise_if_empty=False,
        ).with_columns(
            pl.lit(season).alias("season"),
        )
        for season in seasons
    ]
    return pl.concat(frames, how="diagonal")


def load_dates(seasons: list[str]) -> pl.LazyFrame:
    """Load date data."""
    seasons = list(map(convert_season_to_year, seasons))
    frames = [
        pl.scan_csv(
            DATA_DIR / f"understat/season/{season}/dates.csv",
            try_parse_dates=True,
            raise_if_empty=False,
        ).with_columns(
            pl.lit(season).alias("season"),
        )
        for season in seasons
    ]
    return pl.concat(frames, how="diagonal")


def load_player_ids() -> pl.LazyFrame:
    """Load FPL-to-understat player ID mappings."""
    return pl.scan_csv(
        DATA_DIR / "understat/player_ids.csv",
        try_parse_dates=True,
    )


def load_team_ids() -> pl.LazyFrame:
    """Load FPL-to-understat team ID mappings."""
    return pl.scan_csv(
        DATA_DIR / "understat/team_ids.csv",
        try_parse_dates=True,
    )


def load_fixture_ids(seasons: list[str]) -> pl.LazyFrame:
    """Load FPL-to-understat fixture ID mappings."""
    seasons = list(map(convert_season_to_year, seasons))
    frames = [
        pl.scan_csv(
            DATA_DIR / f"understat/season/{season}/fixture_ids.csv",
            try_parse_dates=True,
            raise_if_empty=False,
        ).with_columns(
            pl.lit(season).alias("season"),
        )
        for season in seasons
    ]
    return pl.concat(frames, how="diagonal")
