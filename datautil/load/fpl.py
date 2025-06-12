import polars as pl

from datautil.constants import DATA_DIR
from datautil.load.fplcache import (
    get_gameweeks,
    load_static_elements,
    load_static_teams,
)
from game.rules import DEF, FWD, GKP, MID, MNG


def load_fpl(seasons: list[str]) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    """Load local FPL data for the given seasons."""

    # Load local data
    elements = load_elements(seasons)
    fixtures = load_fixtures(seasons)

    # Load static data per season and gameweek
    static_teams = pl.concat(
        [
            load_static_teams(season, gameweek)
            for season in seasons
            for gameweek in get_gameweeks(season)
        ],
        how="diagonal_relaxed",
    )
    static_elements = pl.concat(
        [
            load_static_elements(season, gameweek)
            for season in seasons
            for gameweek in get_gameweeks(season)
        ],
        how="diagonal_relaxed",
    )

    # Add "element_type" and "code" to elements
    elements = elements.join(
        static_elements.select(
            [
                pl.col("season"),
                pl.col("round"),
                pl.col("id").alias("element"),
                pl.col("element_type"),
                pl.col("code"),
            ]
        ),
        how="left",
        on=["season", "round", "element"],
    )

    # Add "team" to elements
    elements = elements.join(
        fixtures.select(
            [
                pl.col("id").alias("fixture"),
                pl.col("season"),
                pl.col("team_h"),
                pl.col("team_a"),
            ]
        ),
        how="left",
        on=["season", "fixture"],
    )
    elements = elements.with_columns(
        pl.when(pl.col("was_home"))
        .then(pl.col("team_h"))
        .otherwise(pl.col("team_a"))
        .alias("team")
    )
    elements = elements.drop(["team_h", "team_a"])

    # Add "team_code" and "opponent_team_code" to elements
    elements = elements.join(
        static_teams.select(
            [
                pl.col("season"),
                pl.col("round"),
                pl.col("id").alias("team"),
                pl.col("code").alias("team_code"),
            ]
        ),
        how="left",
        on=["season", "round", "team"],
    )
    elements = elements.join(
        static_teams.select(
            [
                pl.col("season"),
                pl.col("round"),
                pl.col("id").alias("opponent_team"),
                pl.col("code").alias("opponent_team_code"),
            ]
        ),
        how="left",
        on=["season", "round", "opponent_team"],
    )

    # Split elements into players and managers
    players = elements.filter(pl.col("element_type").is_in([GKP, DEF, MID, FWD]))
    managers = elements.filter(pl.col("element_type").is_in([MNG]))

    # Remove manager specific columns from players
    players = players.drop(
        [
            "mng_win",
            "mng_draw",
            "mng_loss",
            "mng_underdog_win",
            "mng_underdog_draw",
            "mng_clean_sheets",
            "mng_goals_scored",
        ],
        # These columns are only available from the 2024-25 season
        strict=False,
    )

    # Map news and availability information to players
    players = players.join(
        static_elements.select(
            [
                "season",
                "round",
                "code",
                "chance_of_playing_next_round",
                "status",
                "news",
                "news_added",
            ]
        ),
        on=["season", "round", "code"],
        how="left",
    )

    # Load team information
    teams = transform_fixtures_to_teams(fixtures, static_teams)

    # Add team codes
    teams = teams.join(
        static_teams.select(
            [
                pl.col("season"),
                pl.col("round"),
                pl.col("id"),
                pl.col("code"),
            ]
        ),
        how="left",
        on=["season", "round", "id"],
    )

    # Add team strength information
    teams = teams.join(
        static_teams.select(
            [
                pl.col("season"),
                pl.col("round"),
                pl.col("id"),
                pl.col("strength"),
                pl.col("strength_attack_home"),
                pl.col("strength_attack_away"),
                pl.col("strength_defence_home"),
                pl.col("strength_defence_away"),
                pl.col("strength_overall_home"),
                pl.col("strength_overall_away"),
            ]
        ),
        how="left",
        on=["season", "round", "id"],
    )

    return players, teams, managers


def load_elements(seasons: list[str]) -> pl.LazyFrame:
    """Load per-gameweek data for all players and managers."""
    frames = [
        pl.scan_csv(
            DATA_DIR / f"api/{season}/players/*.csv",
            try_parse_dates=True,
            raise_if_empty=False,
        ).with_columns(pl.lit(season).alias("season"))
        for season in seasons
    ]
    elements = pl.concat(frames, how="diagonal_relaxed")

    # Remove discontinued features
    elements = elements.drop(
        [
            "attempted_passes",
            "big_chances_created",
            "big_chances_missed",
            "clearances_blocks_interceptions",
            "completed_passes",
            "dribbles",
            "ea_index",
            "errors_leading_to_goal",
            "errors_leading_to_goal_attempt",
            "fouls",
            "id",
            "key_passes",
            "kickoff_time_formatted",
            "loaned_in",
            "loaned_out",
            "offside",
            "open_play_crosses",
            "penalties_conceded",
            "recoveries",
            "tackled",
            "tackles",
            "target_missed",
            "winning_goals",
        ],
        strict=False,
    )
    return elements


def load_fixtures(seasons: list[str]) -> pl.LazyFrame:
    """Load fixture information."""
    frames = [
        pl.scan_csv(
            DATA_DIR / f"api/{season}/fixtures.csv",
            try_parse_dates=True,
            raise_if_empty=False,
        ).with_columns(pl.lit(season).alias("season"))
        for season in seasons
    ]
    fixtures = pl.concat(frames, how="diagonal_relaxed")

    # Only load columns that are available to all seasons
    fixtures = fixtures.select(
        [
            "season",
            "id",
            "event",
            "kickoff_time",
            "team_h",
            "team_a",
            "team_h_score",
            "team_a_score",
        ]
    )

    return fixtures


def transform_fixtures_to_teams(
    fixtures: pl.LazyFrame, static_teams: pl.LazyFrame
) -> pl.LazyFrame:
    """Transform per-match (fixture) data into per-team data."""
    # Select relevant columns from fixtures
    matches = fixtures.select(
        [
            pl.col("season"),
            pl.col("event").alias("round"),
            pl.col("id").alias("fixture_id"),
            pl.col("kickoff_time"),
            pl.col("team_h"),
            pl.col("team_a"),
            pl.col("team_h_score"),
            pl.col("team_a_score"),
        ]
    )
    # Create records for home and away teams
    home_teams = matches.select(
        [
            pl.col("season"),
            pl.col("round"),
            pl.col("fixture_id"),
            pl.col("kickoff_time"),
            pl.col("team_h").alias("id"),
            pl.col("team_a").alias("opponent_id"),
            pl.col("team_h_score").alias("scored"),
            pl.col("team_a_score").alias("conceded"),
            pl.lit(1).alias("was_home"),
        ]
    )
    away_teams = matches.select(
        [
            pl.col("season"),
            pl.col("round"),
            pl.col("fixture_id"),
            pl.col("kickoff_time"),
            pl.col("team_a").alias("id"),
            pl.col("team_h").alias("opponent_id"),
            pl.col("team_a_score").alias("scored"),
            pl.col("team_h_score").alias("conceded"),
            pl.lit(0).alias("was_home"),
        ]
    )
    teams = pl.concat([home_teams, away_teams], how="diagonal_relaxed")
    # Add team and opponent codes
    teams = teams.join(
        static_teams.select(
            [
                pl.col("season"),
                pl.col("round"),
                pl.col("id"),
                pl.col("code").alias("team_code"),
            ]
        ),
        how="left",
        on=["season", "round", "id"],
    )
    teams = teams.join(
        static_teams.select(
            [
                pl.col("season"),
                pl.col("round"),
                pl.col("id").alias("opponent_id"),
                pl.col("code").alias("opponent_code"),
            ]
        ),
        how="left",
        on=["season", "round", "opponent_id"],
    )
    return teams
