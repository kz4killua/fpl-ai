import polars as pl

from .availability import compute_availability
from .relative_strength import compute_relative_strength
from .rolling_mean import compute_rolling_mean


def engineer_player_features(players: pl.LazyFrame) -> pl.LazyFrame:
    return (
        # Compute short-term form
        players.pipe(
            compute_rolling_mean,
            order_by="kickoff_time",
            group_by="code",
            columns=[
                "minutes",
                "total_points",
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
            window_sizes=[3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            defaults=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        )
        # Compute long-term form
        .pipe(
            compute_rolling_mean,
            order_by="kickoff_time",
            group_by="code",
            columns=[
                "minutes",
                "total_points",
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
            window_sizes=[10, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
            defaults=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        )
        .pipe(compute_availability)
    )


def engineer_team_features(teams: pl.LazyFrame) -> pl.LazyFrame:
    return teams.pipe(
        compute_rolling_mean,
        order_by="kickoff_time",
        group_by="code",
        columns=[
            "scored",
            "scored",
            "conceded",
            "conceded",
            "uds_xG",
            "uds_xG",
            "uds_xGA",
            "uds_xGA",
        ],
        window_sizes=[10, 30, 10, 30, 10, 30, 10, 30],
        defaults=[0, 0, 0, 0, 0, 0, 0, 0],
    )


def engineer_match_features(team_features: pl.LazyFrame) -> pl.LazyFrame:
    matches = transform_teams_to_matches(team_features)
    return matches.pipe(compute_relative_strength)


def transform_teams_to_matches(teams: pl.LazyFrame) -> pl.LazyFrame:
    """Transform per-team data into per-match data."""
    columns = [
        "scored_rolling_mean_10",
        "scored_rolling_mean_30",
        "conceded_rolling_mean_10",
        "conceded_rolling_mean_30",
        "uds_xG_rolling_mean_10",
        "uds_xG_rolling_mean_30",
        "uds_xGA_rolling_mean_10",
        "uds_xGA_rolling_mean_30",
        "strength_overall_home",
        "strength_overall_away",
        "strength_attack_home",
        "strength_attack_away",
        "strength_defence_home",
        "strength_defence_away",
    ]
    home_teams = teams.filter(pl.col("was_home") == 1).select(
        [
            pl.col("season"),
            pl.col("fixture_id"),
            pl.col("round"),
            pl.col("code").alias("team_h_code"),
            pl.col("scored").alias("team_h_scored"),
            *[pl.col(column).alias(f"team_h_{column}") for column in columns],
        ]
    )
    away_teams = teams.filter(pl.col("was_home") == 0).select(
        [
            pl.col("season"),
            pl.col("fixture_id"),
            pl.col("round"),
            pl.col("code").alias("team_a_code"),
            pl.col("scored").alias("team_a_scored"),
            *[pl.col(column).alias(f"team_a_{column}") for column in columns],
        ]
    )
    matches = home_teams.join(
        away_teams,
        on=["season", "fixture_id"],
        how="inner",
    )
    return matches
