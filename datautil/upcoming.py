import polars as pl


def get_upcoming_elements(
    season: str,
    gameweeks: list[int],
    fixtures: pl.LazyFrame,
    static_elements: pl.LazyFrame,
) -> pl.LazyFrame:
    upcoming_fixtures = get_upcoming_fixtures(fixtures, season, gameweeks)

    # Select the most recent static teams data
    static_elements = static_elements.filter(
        pl.col("season").eq(season) & pl.col("round").eq(min(gameweeks))
    )

    # Select relevant columns to begin the transformation
    df = upcoming_fixtures.select(
        pl.col("season"),
        pl.col("event").alias("round"),
        pl.col("id").alias("fixture"),
        pl.col("kickoff_time"),
        pl.col("team_a"),
        pl.col("team_h"),
    )

    # Split into records for home and away teams
    df = pl.concat(
        [
            df.with_columns(
                pl.col("team_h").alias("team"),
                pl.col("team_a").alias("opponent_team"),
                pl.lit(True).alias("was_home"),
            ),
            df.with_columns(
                pl.col("team_a").alias("team"),
                pl.col("team_h").alias("opponent_team"),
                pl.lit(False).alias("was_home"),
            ),
        ],
        how="vertical",
    )
    df = df.drop(["team_h", "team_a"])

    # Add elements for each team
    df = df.join(
        static_elements.select(
            pl.col("season"),
            pl.col("team"),
            pl.col("id").alias("element"),
        ),
        on=["season", "team"],
        how="inner",
    )

    # Add values for each element
    df = df.join(
        static_elements.filter(
            pl.col("season").eq(season) & pl.col("round").eq(min(gameweeks))
        ).select(
            pl.col("season"),
            pl.col("id").alias("element"),
            pl.col("now_cost").alias("value"),
        ),
        on=["season", "element"],
        how="left",
    )
    df = df.sort("kickoff_time").with_columns(
        pl.col("value").forward_fill().over(["season", "element"])
    )

    return df


def get_upcoming_static_teams(
    season: str,
    upcoming_gameweeks: list[int],
    static_teams: pl.LazyFrame,
) -> pl.LazyFrame:
    # Select the most recent static teams data
    static_teams = static_teams.filter(
        pl.col("season").eq(season) & pl.col("round").eq(min(upcoming_gameweeks))
    )

    # Keep only columns that we can reasonably assume as constant
    static_teams = static_teams.select(
        "season",
        "round",
        "code",
        "id",
        "name",
        "short_name",
        "pulse_id",
        "strength",
        "strength_attack_away",
        "strength_attack_home",
        "strength_defence_away",
        "strength_defence_home",
        "strength_overall_away",
        "strength_overall_home",
    )

    # Duplicate for each upcoming gameweek
    df = pl.concat(
        [
            static_teams.with_columns(pl.lit(gameweek).alias("round"))
            for gameweek in upcoming_gameweeks
        ],
        how="vertical",
    )

    return df


def get_upcoming_static_elements(
    season: str,
    upcoming_gameweeks: list[int],
    static_elements: pl.LazyFrame,
) -> pl.LazyFrame:
    # Select the most recent static elements data
    static_elements = static_elements.filter(
        pl.col("season").eq(season) & pl.col("round").eq(min(upcoming_gameweeks))
    )

    # Keep only columns that we can reasonably assume as constant
    df = static_elements.select(
        "season",
        "round",
        "code",
        "id",
        "team",
        "team_code",
        "element_type",
        "first_name",
        "second_name",
        "web_name",
        "now_cost",
        "corners_and_indirect_freekicks_order",
        "corners_and_indirect_freekicks_text",
        "direct_freekicks_order",
        "direct_freekicks_text",
        "penalties_order",
        "penalties_text",
    )

    # Duplicate for each upcoming gameweek
    df = pl.concat(
        [
            df.with_columns(pl.lit(gameweek).alias("round"))
            for gameweek in upcoming_gameweeks
        ],
        how="vertical",
    )

    # Add availability information for only the next gameweek
    df = df.join(
        static_elements.select(
            "season",
            "round",
            "code",
            "status",
            "chance_of_playing_next_round",
            "news",
            "news_added",
        ).filter(
            pl.col("season").eq(season) & pl.col("round").eq(min(upcoming_gameweeks))
        ),
        on=["season", "round", "code"],
        how="left",
    )

    return df


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


def remove_upcoming_data(
    df: pl.LazyFrame, season: str, next_gameweek: int
) -> pl.LazyFrame:
    """Remove all records on or after the given gameweek."""
    upcoming = get_upcoming_condition(season, next_gameweek)
    return df.filter(~upcoming)


def mask_upcoming_data(
    df: pl.LazyFrame, season: str, next_gameweek: int, columns: list[str]
) -> pl.LazyFrame:
    """Set upcoming values to `None`."""
    upcoming = get_upcoming_condition(
        season, next_gameweek, gameweek_column="event"
    )

    expressions = []
    for column in columns:
        expressions.append(
            pl.when(upcoming).then(pl.lit(None)).otherwise(pl.col(column)).alias(column)
        )

    return df.with_columns(expressions)


def get_upcoming_condition(
    season: str,
    next_gameweek: int,
    season_column: str = "season",
    gameweek_column: str = "round",
):
    """Return a Polars expression matching upcoming data."""
    return (pl.col(season_column) > season) | (
        (pl.col(season_column) == season)
        & (pl.col(gameweek_column) >= next_gameweek)
    )
