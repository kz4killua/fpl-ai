import polars as pl

from features.balanced_mean import compute_balanced_mean
from features.clb_features import compute_clb_features
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
from features.toa_features import compute_toa_features
from loaders.utils import force_dataframe, get_matches_view, get_teams_view

from .availability import compute_availability
from .last_season_mean import compute_last_season_mean
from .record_count import compute_record_count
from .relative_strength import compute_relative_strength
from .rolling_mean import compute_rolling_mean


def engineer_player_features(df: pl.LazyFrame) -> pl.LazyFrame:
    base_columns = [
        "goals_scored",
        "assists",
        "saves",
        "clearances_blocks_interceptions",
        "tackles",
        "recoveries",
        "influence",
        "creativity",
        "threat",
        "ict_index",
        "uds_xG",
        "uds_xA",
    ]
    base_windows = [3, 5, 10, 20]

    # Create extra features from the base columns
    df = compute_availability(df)
    df = compute_record_count(df, on="total_points")
    df = compute_imputed_set_piece_order(df)
    df = compute_minutes_category(df)
    df = compute_one_hot_minutes_category(df)
    df = compute_per_90(df, base_columns)
    df = compute_share(df, base_columns)

    # Compute rolling means
    rolling_mean_columns = []
    rolling_mean_windows = []

    derived_columns = (
        base_columns
        + [f"{c}_per_90" for c in base_columns]
        + [f"{c}_share" for c in base_columns]
    )

    for c in derived_columns:
        for w in base_windows:
            rolling_mean_columns.append(c)
            rolling_mean_windows.append(w)

    # Handle columns for minutes and availability separately
    minutes_columns = [
        "availability",
        "minutes",
        "minutes_category_0_minutes",
        "minutes_category_1_to_59_minutes",
        "minutes_category_60_plus_minutes",
    ]
    minutes_windows = [1, 3, 5, 10, 20, 38]

    for c in minutes_columns:
        for w in minutes_windows:
            rolling_mean_columns.append(c)
            rolling_mean_windows.append(w)

    df = compute_rolling_mean(df, rolling_mean_columns, rolling_mean_windows)

    # Compute rolling standard deviations (only for minutes)
    rolling_std_columns = []
    rolling_std_windows = []

    for c in minutes_columns:
        for w in [3, 5, 10, 20]:
            rolling_std_columns.append(c)
            rolling_std_windows.append(w)

    df = compute_rolling_std(df, rolling_std_columns, rolling_std_windows)

    # Weight each average using the average of the previous season
    last_season_mean_columns = minutes_columns + derived_columns
    df = compute_last_season_mean(df, last_season_mean_columns)

    for c in derived_columns:
        df = compute_imputed_last_season_mean(df, f"{c}_mean_last_season")

    # Materialize dataframe to avoid OOM issues
    df = force_dataframe(df)

    for c in derived_columns:
        for w in base_windows:
            df = compute_balanced_mean(
                df,
                this_season_column=f"{c}_rolling_mean_{w}",
                last_season_column=f"imputed_{c}_mean_last_season",
                decay=0.7,
                default=0.0,
            )

    # Compute standard deviations over the last season
    last_season_std_columns = minutes_columns
    df = compute_last_season_std(df, last_season_std_columns)

    # Create "_when_available" columns (for minutes)
    available_condition = pl.col("availability") == 100
    df = compute_rolling_mean(
        df,
        rolling_mean_columns,
        rolling_mean_windows,
        condition=available_condition,
        suffix="_when_available",
    )
    df = compute_rolling_std(
        df,
        rolling_std_columns,
        rolling_std_windows,
        condition=available_condition,
        suffix="_when_available",
    )
    df = compute_last_season_mean(
        df,
        last_season_mean_columns,
        condition=available_condition,
        suffix="_when_available",
    )
    df = compute_last_season_std(
        df,
        last_season_std_columns,
        condition=available_condition,
        suffix="_when_available",
    )

    # Compute fatigue
    for w in [5, 7, 10, 14]:
        df = compute_fatigue(df, window=w)

    # Compute features for squad depth
    df = compute_depth_rank(df, "value")
    df = compute_depth_rank(df, "minutes_rolling_mean_38")
    df = compute_depth_unavailability(df, "value")
    df = compute_depth_unavailability(df, "minutes_rolling_mean_38")

    return df


def engineer_match_features(matches: pl.LazyFrame) -> pl.LazyFrame:
    # Compute team level features
    teams = get_teams_view(matches)

    base_columns = ["goals_scored", "goals_conceded", "uds_xG", "uds_xGA"]
    base_windows = [5, 10, 20, 30, 40]

    rolling_mean_columns = []
    rolling_mean_windows = []

    for column in base_columns:
        for window in base_windows:
            rolling_mean_columns.append(column)
            rolling_mean_windows.append(window)

    teams = compute_rolling_mean(
        teams,
        rolling_mean_columns,
        rolling_mean_windows,
        # Compute team averages over just codes
        over=["code"],
    )

    teams = compute_last_season_mean(teams, base_columns)

    # Add match level features
    matches = get_matches_view(teams, extra_fixed_columns=["toa_bookmakers"])
    matches = compute_relative_strength(matches)
    matches = compute_clb_features(matches)

    bookmaker_weights = {
        "pinnacle": 1.0,
        "betfair_ex_uk": 0.5,
        "smarkets": 0.1,
        "skybet": 0.1,
        "matchbook": 0.1,
    }
    matches = compute_toa_features(matches, bookmaker_weights)

    return matches
