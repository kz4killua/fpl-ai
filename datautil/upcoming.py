import polars as pl


def get_upcoming_player_data(
    upcoming_fixtures: pl.LazyFrame,
    static_players: pl.LazyFrame,
    static_teams: pl.LazyFrame,
) -> pl.LazyFrame:
    """Get records for each player to predict upcoming fixtures."""

    # Select relevant columns to begin the merge
    df = upcoming_fixtures.select(
        pl.col("id").alias("fixture"),
        pl.col("event").alias("round"),
        pl.col("kickoff_time"),
        pl.col("season"),
        pl.col("team_a"),
        pl.col("team_h"),
    )

    # Add team identifiers for both home and away teams
    home_df = df.with_columns(
        pl.col("team_h").alias("team"),
        pl.col("team_a").alias("opponent_team"),
        pl.lit(1).alias("was_home"),
    )
    away_df = df.with_columns(
        pl.col("team_a").alias("team"),
        pl.col("team_h").alias("opponent_team"),
        pl.lit(0).alias("was_home"),
    )
    df = pl.concat([home_df, away_df], how="vertical")
    df = df.drop(["team_a", "team_h"])

    # Add team codes and opponent codes
    df = df.join(
        static_teams.select(
            pl.col("id").alias("team"),
            pl.col("code").alias("team_code"),
        ),
        on="team",
        how="left",
    )
    df = df.join(
        static_teams.select(
            pl.col("id").alias("opponent_team"),
            pl.col("code").alias("opponent_team_code"),
        ),
        on="opponent_team",
        how="left",
    )

    # Add players for each team
    df = df.join(
        static_players.select(
            pl.col("team_code"),
            pl.col("id").alias("element"),
            pl.col("code"),
            pl.col("element_type"),
        ),
        on="team_code",
        how="inner",
    )

    # Add known availability information
    df = df.join(
        static_players.select(
            pl.col("season"),
            pl.col("round"),
            pl.col("code"),
            pl.col("status"),
            pl.col("news"),
            pl.col("news_added"),
            pl.col("chance_of_playing_next_round"),
        ),
        on=["season", "round", "code"],
        how="left"
    )

    return df


def get_upcoming_team_data(
    upcoming_fixtures: pl.LazyFrame,
    static_teams: pl.LazyFrame,
) -> pl.LazyFrame:
    """Get records for each team to predict upcoming fixtures."""

    # Select relevant columns to begin the merge
    df = upcoming_fixtures.select(
        pl.col("id").alias("fixture_id"),
        pl.col("event").alias("round"),
        pl.col("kickoff_time"),
        pl.col("season"),
        pl.col("team_a"),
        pl.col("team_h"),
    )

    # Add identifiers for both home and away teams
    home_df = df.with_columns(
        pl.col("team_h").alias("id"),
        pl.col("team_a").alias("opponent_id"),
        pl.lit(1).alias("was_home"),
    )
    away_df = df.with_columns(
        pl.col("team_a").alias("id"),
        pl.col("team_h").alias("opponent_id"),
        pl.lit(0).alias("was_home"),
    )
    df = pl.concat([home_df, away_df], how="vertical")
    df = df.drop(["team_a", "team_h"])

    # Add codes and opponent codes
    df = df.join(
        static_teams.select(
            pl.col("id"),
            pl.col("code"),
        ),
        on="id",
        how="left",
    )
    df = df.join(
        static_teams.select(
            pl.col("id").alias("opponent_id"),
            pl.col("code").alias("opponent_code"),
        ),
        on="opponent_id",
        how="left",
    )

    # Add team strengths
    df = df.join(
        static_teams.select(
            pl.col("code"),
            pl.col("strength"),
            pl.col("strength_overall_home"),
            pl.col("strength_overall_away"),
            pl.col("strength_attack_home"),
            pl.col("strength_attack_away"),
            pl.col("strength_defence_home"),
            pl.col("strength_defence_away"),
        ),
        on="code",
        how="left",
    )

    return df


def get_upcoming_manager_data(
    upcoming_fixtures: pl.LazyFrame,
    static_managers: pl.LazyFrame,
    static_teams: pl.LazyFrame,
) -> pl.LazyFrame:
    """Alias of `get_upcoming_player_data`."""
    return get_upcoming_player_data(
        upcoming_fixtures,
        static_managers,
        static_teams,
    )


def get_upcoming_fixtures(
    fixtures: pl.LazyFrame, season: str, upcoming_gameweeks: list[int]
) -> pl.LazyFrame:
    """Get fixtures for the upcoming gameweeks."""
    df = fixtures.filter(
        (pl.col("event").is_in(upcoming_gameweeks)) & (pl.col("season") == season)
    )
    # Remove upcoming scores as they should be unknown
    df = df.with_columns(
        pl.lit(None).alias("team_a_score"),
        pl.lit(None).alias("team_h_score"),
    )
    return df


def get_upcoming_gameweeks(
    next_gameweek: int, window_size: int, last_gameweek: int
) -> list[int]:
    """Get the list of gameweeks to be optimized for."""
    return list(
        range(next_gameweek, min(next_gameweek + window_size, last_gameweek + 1))
    )
