import polars as pl

from datautil.load.fpl import load_fpl
from datautil.load.understat import load_understat


def load_merged(seasons: list[str]) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    """Load merged player, team, and manager data."""

    # Load data from all sources
    fpl_players, fpl_managers = load_fpl(seasons)
    uds_players, uds_teams = load_understat(seasons)

    # Merge data
    players = merge_players(fpl_players, uds_players)
    teams = uds_teams
    managers = fpl_managers

    # Clean data
    players = clean_players(players)
    teams = clean_teams(teams)

    return players, teams, managers


def merge_players(fpl_players: pl.LazyFrame, uds_players: pl.LazyFrame):
    """Merge player data from FPL and Understat."""
    players = fpl_players.join(
        uds_players.select(
            [
                pl.col("fpl_code"),
                pl.col("fpl_season"),
                pl.col("fpl_fixture_id"),
                pl.col("shots").alias("uds_shots"),
                pl.col("xG").alias("uds_xG"),
                pl.col("position").alias("uds_position"),
                pl.col("xA").alias("uds_xA"),
                pl.col("key_passes").alias("uds_key_passes"),
                pl.col("npg").alias("uds_npg"),
                pl.col("npxG").alias("uds_npxG"),
                pl.col("xGChain").alias("uds_xGChain"),
                pl.col("xGBuildup").alias("uds_xGBuildup"),
            ]
        ),
        how="left",
        left_on=["code", "season", "fixture"],
        right_on=["fpl_code", "fpl_season", "fpl_fixture_id"],
    )
    return players


def merge_teams(fpl_static_teams: pl.LazyFrame, uds_teams: pl.LazyFrame):
    """Merge team data from FPL and Understat."""
    return uds_teams


def clean_players(players: pl.LazyFrame) -> pl.LazyFrame:
    """Clean player data."""

    # Fill in missing values for understat columns (except uds_position)
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
    players = players.with_columns(pl.col("uds_position").fill_null("Reserves"))

    # Cast was_home to an integer
    players = players.with_columns(pl.col("was_home").cast(pl.Int8))

    return players


def clean_teams(teams: pl.LazyFrame) -> pl.LazyFrame:
    """Clean team data."""
    # Cast was_home to an integer
    return teams.with_columns(
        (pl.col("h_a") == "h").cast(pl.Int8).alias("was_home"),
    ).drop(["h_a"])
