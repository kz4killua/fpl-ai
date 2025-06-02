import polars as pl

from .availability import compute_availability
from .rolling_mean import compute_rolling_mean


def engineer_features(players: pl.LazyFrame, teams: pl.LazyFrame) -> pl.LazyFrame:
    """Create prediction features for the model."""

    # Engineer features for players and teams
    players = players.pipe(engineer_player_features)
    teams = teams.pipe(engineer_team_features)

    # Add features for each player's team and the opponent's team
    columns = [
        "uds_xG_rolling_mean_10",
        "uds_xGA_rolling_mean_10",
    ]
    players = players.join(
        teams.select(
            [
                pl.col("season"),
                pl.col("fixture_id").alias("fixture"),
                pl.col("code").alias("team_code"),
                *[pl.col(col).alias(f"team_{col}") for col in columns],
            ]
        ),
        on=["season", "fixture", "team_code"],
        how="left",
    )
    players = players.join(
        teams.select(
            [
                pl.col("season"),
                pl.col("fixture_id").alias("fixture"),
                pl.col("code").alias("opponent_team_code"),
                *[pl.col(col).alias(f"opponent_team_{col}") for col in columns],
            ]
        ),
        on=["season", "fixture", "opponent_team_code"],
        how="left",
    )

    return players


def engineer_player_features(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.pipe(
        compute_rolling_mean,
        order_by="kickoff_time",
        group_by="code",
        columns=[
            "total_points",
            "minutes",
            "uds_xG",
            "uds_xA",
            "clean_sheets",
            "goals_conceded",
            "saves",
            "bonus",
            "influence",
            "creativity",
            "threat",
            "ict_index",
        ],
        window_sizes=[5, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        defaults=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ).pipe(compute_availability)


def engineer_team_features(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.pipe(
        compute_rolling_mean,
        order_by="date",
        group_by="code",
        columns=[
            "uds_xG",
            "uds_xGA",
        ],
        window_sizes=[10, 10],
        defaults=[0, 0],
    )
