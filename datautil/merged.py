from datetime import datetime

import polars as pl

from datautil.clubelo import load_clubelo
from datautil.fpl import load_bootstrap_static, load_fpl
from datautil.understat import load_understat


def load_merged(
    seasons: list[str],
    current_season: str | None = None,
    upcoming_gameweeks: list[int] | None = None,
) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    """Load merged player, team, and manager data."""

    # Get the cutoff time for loaded data
    if upcoming_gameweeks:
        next_gameweek = min(upcoming_gameweeks)
        bootstrap_static = load_bootstrap_static(current_season, next_gameweek)
        for event in bootstrap_static["events"]:
            if event["id"] == next_gameweek:
                cutoff_time = datetime.fromisoformat(event["deadline_time"])
                break
        else:
            raise ValueError(
                f"Could not find cutoff time for gameweek {next_gameweek}."
            )
    else:
        cutoff_time = datetime.max

    # Load data from all sources
    fpl_players, fpl_teams, fpl_managers = load_fpl(
        seasons, current_season, upcoming_gameweeks
    )
    uds_players, uds_teams = load_understat(seasons, cutoff_time)
    clb_teams = load_clubelo(cutoff_time)

    players = merge_players(fpl_players, uds_players, cutoff_time)
    teams = merge_teams(fpl_teams, uds_teams, clb_teams)
    managers = fpl_managers

    return players, teams, managers


def merge_players(
    fpl_players: pl.LazyFrame,
    uds_players: pl.LazyFrame,
    cutoff_time: datetime,
):
    """Merge player data from FPL and Understat."""
    # Map understat attributes to each player
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
    aliases = [f"uds_{column}" for column in columns]

    fpl_players = fpl_players.join(
        uds_players.select(
            [
                pl.col("fpl_code").alias("code"),
                pl.col("fpl_season").alias("season"),
                pl.col("fpl_fixture_id").alias("fixture"),
                *[
                    pl.col(column).alias(alias)
                    for column, alias in zip(columns, aliases, strict=True)
                ],
            ]
        ),
        how="left",
        on=["code", "season", "fixture"],
    )

    # Fill null values, but not for upcoming gameweeks
    expressions = []
    for column in aliases:
        expressions.append(
            pl.when(pl.col("kickoff_time") < cutoff_time)
            .then(pl.col(column).fill_null(0))
            .otherwise(pl.col(column))
            .alias(column)
        )

    return fpl_players


def merge_teams(
    fpl_teams: pl.LazyFrame,
    uds_teams: pl.LazyFrame,
    clb_teams: pl.LazyFrame,
):
    """Merge team data from fpl, understat.com, and clubelo.com"""

    # Map understat attributes to each team
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
    aliases = [f"uds_{column}" for column in columns]

    fpl_teams = fpl_teams.join(
        uds_teams.select(
            pl.col("fpl_season").alias("season"),
            pl.col("fpl_fixture_id").alias("fixture_id"),
            pl.col("fpl_code").alias("code"),
            *[
                pl.col(column).alias(alias)
                for column, alias in zip(columns, aliases, strict=True)
            ],
        ),
        how="left",
        on=["season", "fixture_id", "code"],
    )

    # Add clubelo ratings for teams and opponent teams
    fpl_teams = fpl_teams.sort("kickoff_time")
    clb_teams = clb_teams.sort("To")

    fpl_teams = fpl_teams.join_asof(
        clb_teams.select(
            pl.col("fpl_code").alias("code"),
            pl.col("Elo").alias("clb_elo"),
            pl.col("To").cast(pl.Datetime(time_zone="UTC")),
        ),
        left_on="kickoff_time",
        right_on="To",
        by="code",
        strategy="backward",
        check_sortedness=False,
    ).drop("To")

    fpl_teams = fpl_teams.join_asof(
        clb_teams.select(
            pl.col("fpl_code").alias("opponent_code"),
            pl.col("Elo").alias("opponent_clb_elo"),
            pl.col("To").cast(pl.Datetime(time_zone="UTC")),
        ),
        left_on="kickoff_time",
        right_on="To",
        by="opponent_code",
        strategy="backward",
        check_sortedness=False,
    ).drop("To")

    return fpl_teams
