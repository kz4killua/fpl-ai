import polars as pl


def compute_relative_strength(df: pl.LazyFrame):
    """Compute relative features for teams in a match."""

    expressions = []

    # Calculate expected win probabilities for both sides based on clubelo ratings
    team_h_elo_win_probability = calculate_elo_win_probability(
        pl.col("team_h_clb_elo"), pl.col("team_a_clb_elo")
    ).alias("team_h_clb_elo_win_probability")
    team_a_elo_win_probability = (1 - team_h_elo_win_probability).alias(
        "team_a_clb_elo_win_probability"
    )
    expressions.extend([team_h_elo_win_probability, team_a_elo_win_probability])

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


def calculate_elo_win_probability(team_h_elo: pl.Expr, team_a_elo: pl.Expr) -> pl.Expr:
    """Calculate the win probability for the home team based on clubelo.com ratings."""
    return 1 / (10 ** ((team_a_elo - team_h_elo) / 400) + 1)
