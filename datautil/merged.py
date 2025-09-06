from datetime import UTC, datetime

import polars as pl

from datautil.betting import load_market_probabilities
from datautil.clubelo import load_clubelo
from datautil.fpl import load_bootstrap_static, load_fpl
from datautil.understat import load_understat
from datautil.utils import get_matches_view


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
        next_gameweek = None
        cutoff_time = datetime.max.replace(tzinfo=UTC)

    # Load all data sources
    fpl_players, fpl_teams, fpl_managers = load_fpl(
        seasons, current_season, upcoming_gameweeks
    )
    uds_players, uds_teams = load_understat(seasons, cutoff_time)
    clb_teams = load_clubelo(cutoff_time)
    market_probabilities = load_market_probabilities(
        seasons, cutoff_time, current_season, next_gameweek
    )

    # Merge all data sources
    players = merge_players(fpl_players, uds_players, cutoff_time)
    teams = merge_teams(fpl_teams, uds_teams, clb_teams)
    matches = get_matches_view(teams)
    matches = merge_matches(matches, market_probabilities)
    managers = fpl_managers

    return players, matches, managers


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

    # Add clubelo ratings for teams
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

    return fpl_teams


def merge_matches(matches: pl.LazyFrame, market_probabilities: pl.LazyFrame):
    matches = matches.join(
        market_probabilities.select(
            pl.col("season"),
            pl.col("home_team_code").alias("team_h_code"),
            pl.col("away_team_code").alias("team_a_code"),
            pl.col("home_market_probability"),
            pl.col("away_market_probability"),
            pl.col("draw_market_probability"),
        ),
        on=["season", "team_h_code", "team_a_code"],
        how="left",
    )
    return matches
