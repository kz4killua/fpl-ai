import polars as pl


def compute_relative_strength(df: pl.LazyFrame):
    # Compute relative offensive powers
    team_h_relative_attack_strength = (
        pl.col("team_h_strength_attack_home") / pl.col("team_a_strength_defence_away")
    ).alias("team_h_relative_attack_strength")
    team_a_relative_attack_strength = (
        pl.col("team_a_strength_attack_away") / pl.col("team_h_strength_defence_home")
    ).alias("team_a_relative_attack_strength")

    # Cross rolling means for xG
    team_h_relative_uds_xG_rolling_mean_5 = (
        pl.col("team_h_uds_xG_rolling_mean_5")
        * pl.col("team_a_uds_xGA_rolling_mean_5")
    ).alias("team_h_relative_uds_xG_rolling_mean_5")
    team_a_relative_uds_xG_rolling_mean_5 = (
        pl.col("team_a_uds_xG_rolling_mean_5")
        * pl.col("team_h_uds_xGA_rolling_mean_5")
    ).alias("team_a_relative_uds_xG_rolling_mean_5")

    team_h_relative_uds_xG_rolling_mean_10 = (
        pl.col("team_h_uds_xG_rolling_mean_10")
        * pl.col("team_a_uds_xGA_rolling_mean_10")
    ).alias("team_h_relative_uds_xG_rolling_mean_10")
    team_a_relative_uds_xG_rolling_mean_10 = (
        pl.col("team_a_uds_xG_rolling_mean_10")
        * pl.col("team_h_uds_xGA_rolling_mean_10")
    ).alias("team_a_relative_uds_xG_rolling_mean_10")

    team_h_relative_uds_xG_rolling_mean_20 = (
        pl.col("team_h_uds_xG_rolling_mean_20")
        * pl.col("team_a_uds_xGA_rolling_mean_20")
    ).alias("team_h_relative_uds_xG_rolling_mean_20")
    team_a_relative_uds_xG_rolling_mean_20 = (
        pl.col("team_a_uds_xG_rolling_mean_20")
        * pl.col("team_h_uds_xGA_rolling_mean_20")
    ).alias("team_a_relative_uds_xG_rolling_mean_20")

    team_h_relative_uds_xG_rolling_mean_30 = (
        pl.col("team_h_uds_xG_rolling_mean_30")
        * pl.col("team_a_uds_xGA_rolling_mean_30")
    ).alias("team_h_relative_uds_xG_rolling_mean_30")
    team_a_relative_uds_xG_rolling_mean_30 = (
        pl.col("team_a_uds_xG_rolling_mean_30")
        * pl.col("team_h_uds_xGA_rolling_mean_30")
    ).alias("team_a_relative_uds_xG_rolling_mean_30")

    team_h_relative_uds_xG_rolling_mean_40 = (
        pl.col("team_h_uds_xG_rolling_mean_40")
        * pl.col("team_a_uds_xGA_rolling_mean_40")
    ).alias("team_h_relative_uds_xG_rolling_mean_40")
    team_a_relative_uds_xG_rolling_mean_40 = (
        pl.col("team_a_uds_xG_rolling_mean_40")
        * pl.col("team_h_uds_xGA_rolling_mean_40")
    ).alias("team_a_relative_uds_xG_rolling_mean_40")

    # Cross rolling means for goals scored
    team_h_relative_scored_rolling_mean_5 = (
        pl.col("team_h_scored_rolling_mean_5")
        * pl.col("team_a_conceded_rolling_mean_5")
    ).alias("team_h_relative_scored_rolling_mean_5")
    team_a_relative_scored_rolling_mean_5 = (
        pl.col("team_a_scored_rolling_mean_5")
        * pl.col("team_h_conceded_rolling_mean_5")
    ).alias("team_a_relative_scored_rolling_mean_5")

    team_h_relative_scored_rolling_mean_10 = (
        pl.col("team_h_scored_rolling_mean_10")
        * pl.col("team_a_conceded_rolling_mean_10")
    ).alias("team_h_relative_scored_rolling_mean_10")
    team_a_relative_scored_rolling_mean_10 = (
        pl.col("team_a_scored_rolling_mean_10")
        * pl.col("team_h_conceded_rolling_mean_10")
    ).alias("team_a_relative_scored_rolling_mean_10")

    team_h_relative_scored_rolling_mean_20 = (
        pl.col("team_h_scored_rolling_mean_20")
        * pl.col("team_a_conceded_rolling_mean_20")
    ).alias("team_h_relative_scored_rolling_mean_20")
    team_a_relative_scored_rolling_mean_20 = (
        pl.col("team_a_scored_rolling_mean_20")
        * pl.col("team_h_conceded_rolling_mean_20")
    ).alias("team_a_relative_scored_rolling_mean_20")

    team_h_relative_scored_rolling_mean_30 = (
        pl.col("team_h_scored_rolling_mean_30")
        * pl.col("team_a_conceded_rolling_mean_30")
    ).alias("team_h_relative_scored_rolling_mean_30")
    team_a_relative_scored_rolling_mean_30 = (
        pl.col("team_a_scored_rolling_mean_30")
        * pl.col("team_h_conceded_rolling_mean_30")
    ).alias("team_a_relative_scored_rolling_mean_30")

    team_h_relative_scored_rolling_mean_40 = (
        pl.col("team_h_scored_rolling_mean_40")
        * pl.col("team_a_conceded_rolling_mean_40")
    ).alias("team_h_relative_scored_rolling_mean_40")
    team_a_relative_scored_rolling_mean_40 = (
        pl.col("team_a_scored_rolling_mean_40")
        * pl.col("team_h_conceded_rolling_mean_40")
    ).alias("team_a_relative_scored_rolling_mean_40")

    df = df.with_columns(
        [
            team_h_relative_attack_strength,
            team_a_relative_attack_strength,
            team_h_relative_uds_xG_rolling_mean_5,
            team_a_relative_uds_xG_rolling_mean_5,
            team_h_relative_uds_xG_rolling_mean_10,
            team_a_relative_uds_xG_rolling_mean_10,
            team_h_relative_uds_xG_rolling_mean_20,
            team_a_relative_uds_xG_rolling_mean_20,
            team_h_relative_uds_xG_rolling_mean_30,
            team_a_relative_uds_xG_rolling_mean_30,
            team_h_relative_uds_xG_rolling_mean_40,
            team_a_relative_uds_xG_rolling_mean_40,
            team_h_relative_scored_rolling_mean_5,
            team_a_relative_scored_rolling_mean_5,
            team_h_relative_scored_rolling_mean_10,
            team_a_relative_scored_rolling_mean_10,
            team_h_relative_scored_rolling_mean_20,
            team_a_relative_scored_rolling_mean_20,
            team_h_relative_scored_rolling_mean_30,
            team_a_relative_scored_rolling_mean_30,
            team_h_relative_scored_rolling_mean_40,
            team_a_relative_scored_rolling_mean_40,
        ]
    )

    return df
