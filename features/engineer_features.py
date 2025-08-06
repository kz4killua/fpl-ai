import polars as pl

from features.balanced_mean import compute_balanced_mean
from features.depth_rank import compute_depth_rank
from features.depth_unavailability import compute_depth_unavailability
from features.fatigue import compute_fatigue
from features.imputed_last_season_mean import compute_imputed_last_season_mean
from features.imputed_set_piece_order import compute_imputed_set_piece_order
from features.last_season_std import compute_last_season_std
from features.minutes_category import compute_minutes_category
from features.one_hot_minutes_category import compute_one_hot_minutes_category
from features.per_90 import compute_per_90
from features.rolling_std import compute_rolling_std
from features.share import compute_share

from .availability import compute_availability
from .last_season_mean import compute_last_season_mean
from .record_count import compute_record_count
from .relative_strength import compute_relative_strength
from .rolling_mean import compute_rolling_mean


def engineer_player_features(df: pl.LazyFrame) -> pl.LazyFrame:
    per_90 = [
        # For predicting goals scored
        "uds_xG",
        "goals_scored",
        "threat",
        # For predicting assists
        "uds_xA",
        "assists",
        "creativity",
    ]
    df = compute_per_90(df, columns=per_90)

    share = [
        # For predicting goals scored
        "uds_xG",
        "goals_scored",
        "threat",
        # For predicting assists
        "uds_xA",
        "assists",
        "creativity",
    ]
    df = compute_share(df, columns=share)

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
        ("minutes", 20),
        ("minutes", 38),
        ("minutes_category_1_to_59_minutes", 1),
        ("minutes_category_1_to_59_minutes", 3),
        ("minutes_category_1_to_59_minutes", 5),
        ("minutes_category_1_to_59_minutes", 10),
        ("minutes_category_60_plus_minutes", 1),
        ("minutes_category_60_plus_minutes", 3),
        ("minutes_category_60_plus_minutes", 5),
        ("minutes_category_60_plus_minutes", 10),
        # For predicting goals scored
        ("uds_xG", 3),
        ("uds_xG", 5),
        ("uds_xG", 10),
        ("uds_xG", 20),
        ("uds_xG_share", 3),
        ("uds_xG_share", 5),
        ("uds_xG_share", 10),
        ("uds_xG_share", 20),
        ("goals_scored", 3),
        ("goals_scored", 5),
        ("goals_scored", 10),
        ("goals_scored", 20),
        ("goals_scored_share", 3),
        ("goals_scored_share", 5),
        ("goals_scored_share", 10),
        ("goals_scored_share", 20),
        ("threat", 3),
        ("threat", 5),
        ("threat", 10),
        ("threat", 20),
        ("threat_share", 3),
        ("threat_share", 5),
        ("threat_share", 10),
        ("threat_share", 20),
        ("uds_xG_per_90", 3),
        ("uds_xG_per_90", 5),
        ("uds_xG_per_90", 10),
        ("uds_xG_per_90", 20),
        ("goals_scored_per_90", 3),
        ("goals_scored_per_90", 5),
        ("goals_scored_per_90", 10),
        ("goals_scored_per_90", 20),
        ("threat_per_90", 3),
        ("threat_per_90", 5),
        ("threat_per_90", 10),
        ("threat_per_90", 20),
        # For predicting assists
        ("uds_xA", 3),
        ("uds_xA", 5),
        ("uds_xA", 10),
        ("uds_xA", 20),
        ("uds_xA_share", 3),
        ("uds_xA_share", 5),
        ("uds_xA_share", 10),
        ("uds_xA_share", 20),
        ("assists", 3),
        ("assists", 5),
        ("assists", 10),
        ("assists", 20),
        ("assists_share", 3),
        ("assists_share", 5),
        ("assists_share", 10),
        ("assists_share", 20),
        ("creativity", 3),
        ("creativity", 5),
        ("creativity", 10),
        ("creativity", 20),
        ("creativity_share", 3),
        ("creativity_share", 5),
        ("creativity_share", 10),
        ("creativity_share", 20),
        ("uds_xA_per_90", 3),
        ("uds_xA_per_90", 5),
        ("uds_xA_per_90", 10),
        ("uds_xA_per_90", 20),
        ("assists_per_90", 3),
        ("assists_per_90", 5),
        ("assists_per_90", 10),
        ("assists_per_90", 20),
        ("creativity_per_90", 3),
        ("creativity_per_90", 5),
        ("creativity_per_90", 10),
        ("creativity_per_90", 20),
        # For predicting goals scored and assists
        ("opponent_team_strength_defence_condition", 3),
        ("opponent_team_strength_defence_condition", 5),
        ("opponent_team_strength_defence_condition", 10),
        ("opponent_team_strength_defence_condition", 20),
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
        # For predicting goals scored
        "uds_xG",
        "uds_xG_per_90",
        "uds_xG_share",
        "goals_scored",
        "goals_scored_per_90",
        "goals_scored_share",
        "threat",
        "threat_per_90",
        "threat_share",
        # For predicting assists
        "uds_xA",
        "uds_xA_per_90",
        "uds_xA_share",
        "assists",
        "assists_per_90",
        "assists_share",
        "creativity",
        "creativity_per_90",
        "creativity_share",
    ]

    balanced_means = [
        # For predicting goals scored
        ("uds_xG_rolling_mean_3", "imputed_uds_xG_mean_last_season"),
        ("uds_xG_rolling_mean_5", "imputed_uds_xG_mean_last_season"),
        ("uds_xG_rolling_mean_10", "imputed_uds_xG_mean_last_season"),
        ("uds_xG_rolling_mean_20", "imputed_uds_xG_mean_last_season"),
        ("goals_scored_rolling_mean_3", "imputed_goals_scored_mean_last_season"),
        ("goals_scored_rolling_mean_5", "imputed_goals_scored_mean_last_season"),
        ("goals_scored_rolling_mean_10", "imputed_goals_scored_mean_last_season"),
        ("goals_scored_rolling_mean_20", "imputed_goals_scored_mean_last_season"),
        ("threat_rolling_mean_3", "imputed_threat_mean_last_season"),
        ("threat_rolling_mean_5", "imputed_threat_mean_last_season"),
        ("threat_rolling_mean_10", "imputed_threat_mean_last_season"),
        ("threat_rolling_mean_20", "imputed_threat_mean_last_season"),
        ("uds_xG_per_90_rolling_mean_3", "imputed_uds_xG_per_90_mean_last_season"),
        ("uds_xG_per_90_rolling_mean_5", "imputed_uds_xG_per_90_mean_last_season"),
        ("uds_xG_per_90_rolling_mean_10", "imputed_uds_xG_per_90_mean_last_season"),
        ("uds_xG_per_90_rolling_mean_20", "imputed_uds_xG_per_90_mean_last_season"),
        (
            "goals_scored_per_90_rolling_mean_3",
            "imputed_goals_scored_per_90_mean_last_season",
        ),
        (
            "goals_scored_per_90_rolling_mean_5",
            "imputed_goals_scored_per_90_mean_last_season",
        ),
        (
            "goals_scored_per_90_rolling_mean_10",
            "imputed_goals_scored_per_90_mean_last_season",
        ),
        (
            "goals_scored_per_90_rolling_mean_20",
            "imputed_goals_scored_per_90_mean_last_season",
        ),
        ("threat_per_90_rolling_mean_3", "imputed_threat_per_90_mean_last_season"),
        ("threat_per_90_rolling_mean_5", "imputed_threat_per_90_mean_last_season"),
        ("threat_per_90_rolling_mean_10", "imputed_threat_per_90_mean_last_season"),
        ("threat_per_90_rolling_mean_20", "imputed_threat_per_90_mean_last_season"),
        ("uds_xG_share_rolling_mean_3", "imputed_uds_xG_share_mean_last_season"),
        ("uds_xG_share_rolling_mean_5", "imputed_uds_xG_share_mean_last_season"),
        ("uds_xG_share_rolling_mean_10", "imputed_uds_xG_share_mean_last_season"),
        ("uds_xG_share_rolling_mean_20", "imputed_uds_xG_share_mean_last_season"),
        (
            "goals_scored_share_rolling_mean_3",
            "imputed_goals_scored_share_mean_last_season",
        ),
        (
            "goals_scored_share_rolling_mean_5",
            "imputed_goals_scored_share_mean_last_season",
        ),
        (
            "goals_scored_share_rolling_mean_10",
            "imputed_goals_scored_share_mean_last_season",
        ),
        (
            "goals_scored_share_rolling_mean_20",
            "imputed_goals_scored_share_mean_last_season",
        ),
        ("threat_share_rolling_mean_3", "imputed_threat_share_mean_last_season"),
        ("threat_share_rolling_mean_5", "imputed_threat_share_mean_last_season"),
        ("threat_share_rolling_mean_10", "imputed_threat_share_mean_last_season"),
        ("threat_share_rolling_mean_20", "imputed_threat_share_mean_last_season"),
        # For predicting assists
        ("uds_xA_rolling_mean_3", "imputed_uds_xA_mean_last_season"),
        ("uds_xA_rolling_mean_5", "imputed_uds_xA_mean_last_season"),
        ("uds_xA_rolling_mean_10", "imputed_uds_xA_mean_last_season"),
        ("uds_xA_rolling_mean_20", "imputed_uds_xA_mean_last_season"),
        ("assists_rolling_mean_3", "imputed_assists_mean_last_season"),
        ("assists_rolling_mean_5", "imputed_assists_mean_last_season"),
        ("assists_rolling_mean_10", "imputed_assists_mean_last_season"),
        ("assists_rolling_mean_20", "imputed_assists_mean_last_season"),
        ("creativity_rolling_mean_3", "imputed_creativity_mean_last_season"),
        ("creativity_rolling_mean_5", "imputed_creativity_mean_last_season"),
        ("creativity_rolling_mean_10", "imputed_creativity_mean_last_season"),
        ("creativity_rolling_mean_20", "imputed_creativity_mean_last_season"),
        (
            "creativity_per_90_rolling_mean_3",
            "imputed_creativity_per_90_mean_last_season",
        ),
        (
            "creativity_per_90_rolling_mean_5",
            "imputed_creativity_per_90_mean_last_season",
        ),
        (
            "creativity_per_90_rolling_mean_10",
            "imputed_creativity_per_90_mean_last_season",
        ),
        (
            "creativity_per_90_rolling_mean_20",
            "imputed_creativity_per_90_mean_last_season",
        ),
        (
            "uds_xA_per_90_rolling_mean_3",
            "imputed_uds_xA_per_90_mean_last_season",
        ),
        (
            "uds_xA_per_90_rolling_mean_5",
            "imputed_uds_xA_per_90_mean_last_season",
        ),
        (
            "uds_xA_per_90_rolling_mean_10",
            "imputed_uds_xA_per_90_mean_last_season",
        ),
        (
            "uds_xA_per_90_rolling_mean_20",
            "imputed_uds_xA_per_90_mean_last_season",
        ),
        (
            "assists_per_90_rolling_mean_3",
            "imputed_assists_per_90_mean_last_season",
        ),
        (
            "assists_per_90_rolling_mean_5",
            "imputed_assists_per_90_mean_last_season",
        ),
        (
            "assists_per_90_rolling_mean_10",
            "imputed_assists_per_90_mean_last_season",
        ),
        (
            "assists_per_90_rolling_mean_20",
            "imputed_assists_per_90_mean_last_season",
        ),
        ("uds_xA_share_rolling_mean_3", "imputed_uds_xA_share_mean_last_season"),
        ("uds_xA_share_rolling_mean_5", "imputed_uds_xA_share_mean_last_season"),
        ("uds_xA_share_rolling_mean_10", "imputed_uds_xA_share_mean_last_season"),
        ("uds_xA_share_rolling_mean_20", "imputed_uds_xA_share_mean_last_season"),
        (
            "assists_share_rolling_mean_3",
            "imputed_assists_share_mean_last_season",
        ),
        (
            "assists_share_rolling_mean_5",
            "imputed_assists_share_mean_last_season",
        ),
        (
            "assists_share_rolling_mean_10",
            "imputed_assists_share_mean_last_season",
        ),
        (
            "assists_share_rolling_mean_20",
            "imputed_assists_share_mean_last_season",
        ),
        (
            "creativity_share_rolling_mean_3",
            "imputed_creativity_share_mean_last_season",
        ),
        (
            "creativity_share_rolling_mean_5",
            "imputed_creativity_share_mean_last_season",
        ),
        (
            "creativity_share_rolling_mean_10",
            "imputed_creativity_share_mean_last_season",
        ),
        (
            "creativity_share_rolling_mean_20",
            "imputed_creativity_share_mean_last_season",
        ),
    ]

    imputed_last_season_means = [
        # For predicting goals scored
        "uds_xG_mean_last_season",
        "uds_xG_per_90_mean_last_season",
        "uds_xG_share_mean_last_season",
        "goals_scored_mean_last_season",
        "goals_scored_per_90_mean_last_season",
        "goals_scored_share_mean_last_season",
        "threat_mean_last_season",
        "threat_per_90_mean_last_season",
        "threat_share_mean_last_season",
        # For predicting assists
        "uds_xA_mean_last_season",
        "uds_xA_per_90_mean_last_season",
        "uds_xA_share_mean_last_season",
        "assists_mean_last_season",
        "assists_per_90_mean_last_season",
        "assists_share_mean_last_season",
        "creativity_mean_last_season",
        "creativity_per_90_mean_last_season",
        "creativity_share_mean_last_season",
    ]

    last_season_stds = [
        # For predicting minutes
        "availability",
        "minutes",
        "minutes_category_1_to_59_minutes",
        "minutes_category_60_plus_minutes",
    ]

    rolling_means_when_available = [
        # For predicting minutes
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
        # For predicting goals scored
        ("uds_xG", 3),
        ("uds_xG", 5),
        ("uds_xG", 10),
        ("uds_xG", 20),
        ("goals_scored", 3),
        ("goals_scored", 5),
        ("goals_scored", 10),
        ("goals_scored", 20),
        ("threat", 3),
        ("threat", 5),
        ("threat", 10),
        ("threat", 20),
        # For predicting assists
        ("uds_xA", 3),
        ("uds_xA", 5),
        ("uds_xA", 10),
        ("uds_xA", 20),
        ("assists", 3),
        ("assists", 5),
        ("assists", 10),
        ("assists", 20),
        ("creativity", 3),
        ("creativity", 5),
        ("creativity", 10),
        ("creativity", 20),
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

    df = (
        df.pipe(compute_availability)
        .pipe(compute_record_count, on="total_points")
        .pipe(compute_imputed_set_piece_order)
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
                "total_points",
                "clean_sheets",
                "goals_conceded",
                "saves",
                "bonus",
                "bps",
                "influence",
                "ict_index",
            ],
        )
        # Compute short-term form
        .pipe(
            compute_rolling_mean,
            columns=[
                "total_points",
                "clean_sheets",
                "goals_conceded",
                "saves",
                "bonus",
                "bps",
                "influence",
                "ict_index",
            ],
            window_sizes=[5, 5, 5, 5, 5, 5, 5, 5],
            # Compute player averages over seasons and codes
            over=["season", "code"],
        )
        # Compute long-term form
        .pipe(
            compute_rolling_mean,
            columns=[
                "total_points",
                "clean_sheets",
                "goals_conceded",
                "saves",
                "bonus",
                "bps",
                "influence",
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
            ],
            # Compute player averages over seasons and codes
            over=["season", "code"],
        )
    )

    # Compute imputed last season means
    for column in imputed_last_season_means:
        df = compute_imputed_last_season_mean(df, column)

    # Compute balanced means
    for this_season_column, last_season_column in balanced_means:
        df = compute_balanced_mean(
            df, this_season_column, last_season_column, decay=0.7, default=0.0
        )

    return df


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
        if ("rolling_mean" in column) or ("strength" in column) or ("clb_elo" in column)
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
