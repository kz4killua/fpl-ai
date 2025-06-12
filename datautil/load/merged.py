import polars as pl

from datautil.load.fpl import load_fpl
from datautil.load.understat import load_understat


def load_merged(seasons: list[str]) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    """Load merged player, team, and manager data."""

    # Load data
    fpl_players, fpl_teams, fpl_managers = load_fpl(seasons)
    uds_players, uds_teams = load_understat(seasons)

    # Merge data
    players = merge_players(fpl_players, uds_players)
    teams = merge_teams(fpl_teams, uds_teams)
    managers = fpl_managers

    # Clean data
    players = clean_players(players)
    teams = clean_teams(teams)

    return players, teams, managers


def merge_players(fpl_players: pl.LazyFrame, uds_players: pl.LazyFrame):
    """Merge player data from FPL and Understat."""
    columns = [
        "xG",
        "xA",
        "shots",
        "key_passes",
        "npg",
        "npxG",
        "xGChain",
        "xGBuildup",
        "position",
    ]
    players = fpl_players.join(
        uds_players.select(
            [
                pl.col("fpl_code").alias("code"),
                pl.col("fpl_season").alias("season"),
                pl.col("fpl_fixture_id").alias("fixture"),
                *[pl.col(column).alias(f"uds_{column}") for column in columns],
            ]
        ),
        how="left",
        on=["code", "season", "fixture"],
    )
    return players


def merge_teams(fpl_teams: pl.LazyFrame, uds_teams: pl.LazyFrame):
    """Merge team data from FPL and Understat."""
    columns = [
        "xG",
        "xGA",
        "npxG",
        "npxGA",
        "deep",
        "deep_allowed",
        "missed",
        "xpts",
        "result",
        "wins",
        "draws",
        "loses",
        "pts",
        "npxGD",
        "ppda_att",
        "ppda_def",
        "ppda_allowed_att",
        "ppda_allowed_def",
    ]
    fpl_teams = fpl_teams.join(
        uds_teams.select(
            pl.col("fpl_season").alias("season"),
            pl.col("fpl_fixture_id").alias("fixture_id"),
            pl.col("fpl_code").alias("code"),
            *[pl.col(column).alias(f"uds_{column}") for column in columns],
        ),
        how="left",
        on=["season", "fixture_id", "code"],
    )
    return fpl_teams


def clean_players(players: pl.LazyFrame) -> pl.LazyFrame:
    """Clean player data."""

    # Fill in missing values for numeric understats
    players = players.with_columns(
        pl.col("uds_shots").fill_null(0),
        pl.col("uds_xG").fill_null(0),
        pl.col("uds_xA").fill_null(0),
        pl.col("uds_key_passes").fill_null(0),
        pl.col("uds_npg").fill_null(0),
        pl.col("uds_npxG").fill_null(0),
        pl.col("uds_xGChain").fill_null(0),
        pl.col("uds_xGBuildup").fill_null(0),
    )

    # Fill in missing values for uds_position
    players = players.with_columns(pl.col("uds_position").fill_null("Reserve"))

    # Cast was_home to an integer
    players = players.with_columns(pl.col("was_home").cast(pl.Int8))

    return players


def clean_teams(teams: pl.LazyFrame) -> pl.LazyFrame:
    """Clean team data."""
    return teams
