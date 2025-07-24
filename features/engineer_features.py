import polars as pl

from features.depth_rank import compute_depth_rank
from features.depth_unavailability import compute_depth_unavailability
from features.fatigue import compute_fatigue
from features.last_season_std import compute_last_season_std
from features.minutes_category import compute_minutes_category
from features.one_hot_minutes_category import compute_one_hot_minutes_category
from features.rolling_std import compute_rolling_std

from .availability import compute_availability
from .last_season_mean import compute_last_season_mean
from .record_count import compute_record_count
from .relative_strength import compute_relative_strength
from .rolling_mean import compute_rolling_mean


def engineer_player_features(players: pl.LazyFrame) -> pl.LazyFrame:
    rolling_means = [
        # For predicting minutes
        ("availability", 1),
        ("availability", 3),
        ("availability", 5),
        ("availability", 10),
        ("minutes", 1),
        ("minutes", 3),
        ("minutes", 5),
        ("minutes", 10),
        ("minutes", 38),
        ("minutes_category_1_to_59_minutes", 1),
        ("minutes_category_1_to_59_minutes", 3),
        ("minutes_category_1_to_59_minutes", 5),
        ("minutes_category_1_to_59_minutes", 10),
        ("minutes_category_60_plus_minutes", 1),
        ("minutes_category_60_plus_minutes", 3),
        ("minutes_category_60_plus_minutes", 5),
        ("minutes_category_60_plus_minutes", 10),
    ]

    rolling_stds = [
        # For predicting minutes
        ("availability", 3),
        ("availability", 5),
        ("availability", 10),
        ("minutes", 3),
        ("minutes", 5),
        ("minutes", 10),
        ("minutes_category_1_to_59_minutes", 3),
        ("minutes_category_1_to_59_minutes", 5),
        ("minutes_category_1_to_59_minutes", 10),
        ("minutes_category_60_plus_minutes", 3),
        ("minutes_category_60_plus_minutes", 5),
        ("minutes_category_60_plus_minutes", 10),
    ]

    last_season_means = [
        # For predicting minutes
        "availability",
        "minutes",
        "minutes_category_1_to_59_minutes",
        "minutes_category_60_plus_minutes",
    ]

    last_season_stds = [
        # For predicting minutes
        "availability",
        "minutes",
        "minutes_category_1_to_59_minutes",
        "minutes_category_60_plus_minutes",
    ]

    # For predicting minutes
    rolling_means_when_available = [
        ("minutes", 1),
        ("minutes", 3),
        ("minutes", 5),
        ("minutes", 10),
        ("minutes_category_1_to_59_minutes", 1),
        ("minutes_category_1_to_59_minutes", 3),
        ("minutes_category_1_to_59_minutes", 5),
        ("minutes_category_1_to_59_minutes", 10),
        ("minutes_category_60_plus_minutes", 1),
        ("minutes_category_60_plus_minutes", 3),
        ("minutes_category_60_plus_minutes", 5),
        ("minutes_category_60_plus_minutes", 10),
    ]

    # For predicting minutes
    rolling_stds_when_available = [
        ("minutes", 3),
        ("minutes", 5),
        ("minutes", 10),
        ("minutes_category_1_to_59_minutes", 3),
        ("minutes_category_1_to_59_minutes", 5),
        ("minutes_category_1_to_59_minutes", 10),
        ("minutes_category_60_plus_minutes", 3),
        ("minutes_category_60_plus_minutes", 5),
        ("minutes_category_60_plus_minutes", 10),
    ]

    last_season_means_when_available = last_season_means
    last_season_stds_when_available = last_season_stds

    return (
        players.pipe(compute_availability)
        .pipe(compute_record_count, on="total_points")
        .pipe(compute_minutes_category)
        .pipe(compute_one_hot_minutes_category)
        .pipe(
            compute_rolling_mean,
            columns=[c for c, _ in rolling_means],
            window_sizes=[w for _, w in rolling_means],
        )
        .pipe(
            compute_rolling_std,
            columns=[c for c, _ in rolling_stds],
            window_sizes=[w for _, w in rolling_stds],
        )
        .pipe(
            compute_last_season_mean,
            columns=last_season_means,
        )
        .pipe(
            compute_last_season_std,
            columns=last_season_stds,
        )
        .pipe(
            compute_rolling_mean,
            columns=[c for c, _ in rolling_means_when_available],
            window_sizes=[w for _, w in rolling_means_when_available],
            condition=pl.col("availability") == 100,
            suffix="_when_available",
        )
        .pipe(
            compute_rolling_std,
            columns=[c for c, _ in rolling_stds_when_available],
            window_sizes=[w for _, w in rolling_stds_when_available],
            condition=pl.col("availability") == 100,
            suffix="_when_available",
        )
        .pipe(
            compute_last_season_mean,
            columns=last_season_means_when_available,
            condition=pl.col("availability") == 100,
            suffix="_when_available",
        )
        .pipe(
            compute_last_season_std,
            columns=last_season_stds_when_available,
            condition=pl.col("availability") == 100,
            suffix="_when_available",
        )
        .pipe(compute_fatigue, window=5)
        .pipe(compute_fatigue, window=7)
        .pipe(compute_fatigue, window=10)
        .pipe(compute_fatigue, window=14)
        .pipe(compute_depth_rank, "value")
        .pipe(compute_depth_rank, "minutes_rolling_mean_38")
        .pipe(compute_depth_unavailability, "value")
        .pipe(compute_depth_unavailability, "minutes_rolling_mean_38")
        .pipe(
            compute_last_season_mean,
            columns=[
                "minutes",
                "starts",
                "total_points",
                "goals_scored",
                "assists",
                "uds_xG",
                "uds_xA",
                "clean_sheets",
                "goals_conceded",
                "saves",
                "bonus",
                "bps",
                "influence",
                "creativity",
                "threat",
                "ict_index",
            ],
        )
        # Compute short-term form
        .pipe(
            compute_rolling_mean,
            columns=[
                "total_points",
                "goals_scored",
                "assists",
                "uds_xG",
                "uds_xA",
                "clean_sheets",
                "goals_conceded",
                "saves",
                "bonus",
                "bps",
                "influence",
                "creativity",
                "threat",
                "ict_index",
            ],
            window_sizes=[5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            # Compute player averages over seasons and codes
            over=["season", "code"],
        )
        # Compute long-term form
        .pipe(
            compute_rolling_mean,
            columns=[
                "total_points",
                "goals_scored",
                "assists",
                "uds_xG",
                "uds_xA",
                "clean_sheets",
                "goals_conceded",
                "saves",
                "bonus",
                "bps",
                "influence",
                "creativity",
                "threat",
                "ict_index",
            ],
            window_sizes=[
                20,
                20,
                20,
                20,
                20,
                20,
                20,
                20,
                20,
                20,
                20,
                20,
                20,
                20,
            ],
            # Compute player averages over seasons and codes
            over=["season", "code"],
        )
    )


def engineer_team_features(teams: pl.LazyFrame) -> pl.LazyFrame:
    rolling_means = [
        ("scored", 5),
        ("scored", 10),
        ("scored", 20),
        ("scored", 30),
        ("scored", 40),
        ("conceded", 5),
        ("conceded", 10),
        ("conceded", 20),
        ("conceded", 30),
        ("conceded", 40),
        ("uds_xG", 5),
        ("uds_xG", 10),
        ("uds_xG", 20),
        ("uds_xG", 30),
        ("uds_xG", 40),
        ("uds_xGA", 5),
        ("uds_xGA", 10),
        ("uds_xGA", 20),
        ("uds_xGA", 30),
        ("uds_xGA", 40),
    ]

    last_season_means = [
        "scored",
        "conceded",
        "uds_xG",
        "uds_xGA",
    ]

    return teams.pipe(
        compute_rolling_mean,
        columns=[c for c, _ in rolling_means],
        window_sizes=[w for _, w in rolling_means],
        # Compute team averages over just codes
        over=["code"],
    ).pipe(
        compute_last_season_mean,
        columns=last_season_means,
    )


def engineer_match_features(team_features: pl.LazyFrame) -> pl.LazyFrame:
    matches = transform_teams_to_matches(team_features)
    return matches.pipe(compute_relative_strength)


def transform_teams_to_matches(teams: pl.LazyFrame) -> pl.LazyFrame:
    """Transform per-team data into per-match data."""
    columns = [
        column
        for column in teams.columns
        if ("rolling_mean_" in column) or ("strength_" in column)
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
