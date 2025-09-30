import polars as pl


def compute_relative_strength(df: pl.LazyFrame):
    """Compute relative features for teams in a match."""

    expressions = []

    # Compare each team's attack strength with the opponent's defense strength
    team_h_relative_strength_attack = (
        pl.col("team_h_strength_attack_home") / pl.col("team_a_strength_defence_away")
    ).alias("team_h_relative_strength_attack")
    team_a_relative_strength_attack = (
        pl.col("team_a_strength_attack_away") / pl.col("team_h_strength_defence_home")
    ).alias("team_a_relative_strength_attack")
    expressions.extend(
        [team_h_relative_strength_attack, team_a_relative_strength_attack]
    )

    # Compare each team's xG performance with the opponent's xGA performance
    for window in [5, 10, 20, 40]:
        team_h_relative_uds_xG_rolling_mean = (
            pl.col(f"team_h_uds_xG_rolling_mean_{window}")
            * pl.col(f"team_a_uds_xGA_rolling_mean_{window}")
        ).alias(f"team_h_relative_uds_xG_rolling_mean_{window}")
        team_a_relative_uds_xG_rolling_mean = (
            pl.col(f"team_a_uds_xG_rolling_mean_{window}")
            * pl.col(f"team_h_uds_xGA_rolling_mean_{window}")
        ).alias(f"team_a_relative_uds_xG_rolling_mean_{window}")
        expressions.extend(
            [
                team_h_relative_uds_xG_rolling_mean,
                team_a_relative_uds_xG_rolling_mean,
            ]
        )

    # Compare each team's goals scored with the opponent's goals conceded
    for window in [5, 10, 20, 40]:
        team_h_relative_scored_rolling_mean = (
            pl.col(f"team_h_scored_rolling_mean_{window}")
            * pl.col(f"team_a_conceded_rolling_mean_{window}")
        ).alias(f"team_h_relative_scored_rolling_mean_{window}")
        team_a_relative_scored_rolling_mean = (
            pl.col(f"team_a_scored_rolling_mean_{window}")
            * pl.col(f"team_h_conceded_rolling_mean_{window}")
        ).alias(f"team_a_relative_scored_rolling_mean_{window}")
        expressions.extend(
            [
                team_h_relative_scored_rolling_mean,
                team_a_relative_scored_rolling_mean,
            ]
        )

    return df.with_columns(expressions)
