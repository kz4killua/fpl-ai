import glob
import json
import lzma

import polars as pl

from datautil.constants import DATA_DIR
from datautil.upcoming import (
    get_upcoming_elements,
    get_upcoming_static_elements,
    get_upcoming_static_teams,
    remove_upcoming_data,
)
from game.rules import DEF, FWD, GKP, MID, MNG


def load_fpl(
    seasons: list[str],
    current_season: str | None = None,
    upcoming_gameweeks: list[str] | None = None,
) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    """Load local FPL data for the given seasons."""

    # Load local data
    fixtures = load_fixtures(seasons)
    elements = load_elements(seasons)

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

    if current_season and upcoming_gameweeks:
        if current_season not in seasons:
            raise ValueError(f"Season '{current_season}' is not loaded.")

        # Create records for upcoming gameweeks
        upcoming_elements = get_upcoming_elements(
            current_season, upcoming_gameweeks, fixtures, static_elements
        )
        upcoming_static_teams = get_upcoming_static_teams(
            current_season, upcoming_gameweeks, static_teams
        )
        upcoming_static_elements = get_upcoming_static_elements(
            current_season, upcoming_gameweeks, static_elements
        )

        # Remove already loaded upcoming records
        elements = remove_upcoming_data(
            elements, current_season, min(upcoming_gameweeks)
        )
        static_elements = remove_upcoming_data(
            static_elements, current_season, min(upcoming_gameweeks)
        )
        static_teams = remove_upcoming_data(
            static_teams, current_season, min(upcoming_gameweeks)
        )

        # Add created upcoming records
        elements = pl.concat([elements, upcoming_elements], how="diagonal_relaxed")
        static_elements = pl.concat(
            [static_elements, upcoming_static_elements], how="diagonal_relaxed"
        )
        static_teams = pl.concat(
            [static_teams, upcoming_static_teams], how="diagonal_relaxed"
        )

        # Filter fixtures to only include upcoming gameweeks
        fixtures = fixtures.filter(
            (pl.col("season") < current_season)
            | (
                (pl.col("season") == current_season)
                & (pl.col("event").is_in(upcoming_gameweeks))
            )
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

    # Map availability and set piece information to players
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
                "corners_and_indirect_freekicks_order",
                "direct_freekicks_order",
                "penalties_order",
            ]
        ),
        on=["season", "round", "code"],
        how="left",
    )

    # Add team and opponent strength information to players
    players = players.join(
        static_teams.select(
            [
                pl.col("season"),
                pl.col("round"),
                pl.col("id").alias("team"),
                pl.col("strength").alias("team_strength"),
                pl.col("strength_attack_home").alias("team_strength_attack_home"),
                pl.col("strength_attack_away").alias("team_strength_attack_away"),
                pl.col("strength_defence_home").alias("team_strength_defence_home"),
                pl.col("strength_defence_away").alias("team_strength_defence_away"),
                pl.col("strength_overall_home").alias("team_strength_overall_home"),
                pl.col("strength_overall_away").alias("team_strength_overall_away"),
            ]
        ),
        how="left",
        on=["season", "round", "team"],
    )
    players = players.join(
        static_teams.select(
            [
                pl.col("season"),
                pl.col("round"),
                pl.col("id").alias("opponent_team"),
                pl.col("strength").alias("opponent_team_strength"),
                pl.col("strength_attack_home").alias(
                    "opponent_team_strength_attack_home"
                ),
                pl.col("strength_attack_away").alias(
                    "opponent_team_strength_attack_away"
                ),
                pl.col("strength_defence_home").alias(
                    "opponent_team_strength_defence_home"
                ),
                pl.col("strength_defence_away").alias(
                    "opponent_team_strength_defence_away"
                ),
                pl.col("strength_overall_home").alias(
                    "opponent_team_strength_overall_home"
                ),
                pl.col("strength_overall_away").alias(
                    "opponent_team_strength_overall_away"
                ),
            ]
        ),
        how="left",
        on=["season", "round", "opponent_team"],
    )
    players = players.with_columns(
        # Select team strengths based on home / away status
        pl.when(pl.col("was_home"))
        .then(pl.col("team_strength_attack_home"))
        .otherwise(pl.col("team_strength_attack_away"))
        .alias("team_strength_attack_condition"),
        pl.when(pl.col("was_home"))
        .then(pl.col("team_strength_defence_home"))
        .otherwise(pl.col("team_strength_defence_away"))
        .alias("team_strength_defence_condition"),
        pl.when(pl.col("was_home"))
        .then(pl.col("team_strength_overall_home"))
        .otherwise(pl.col("team_strength_overall_away"))
        .alias("team_strength_overall_condition"),
        # Select opponent team strengths based on home / away status
        pl.when(pl.col("was_home"))
        .then(pl.col("opponent_team_strength_attack_away"))
        .otherwise(pl.col("opponent_team_strength_attack_home"))
        .alias("opponent_team_strength_attack_condition"),
        pl.when(pl.col("was_home"))
        .then(pl.col("opponent_team_strength_defence_away"))
        .otherwise(pl.col("opponent_team_strength_defence_home"))
        .alias("opponent_team_strength_defence_condition"),
        pl.when(pl.col("was_home"))
        .then(pl.col("opponent_team_strength_overall_away"))
        .otherwise(pl.col("opponent_team_strength_overall_home"))
        .alias("opponent_team_strength_overall_condition"),
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

    # Add opponent team codes
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

    # Add team strength information to teams
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

    # Add opponent team strength information to teams
    teams = teams.join(
        static_teams.select(
            [
                pl.col("season"),
                pl.col("round"),
                pl.col("id").alias("opponent_id"),
                pl.col("strength").alias("opponent_strength"),
                pl.col("strength_attack_home").alias("opponent_strength_attack_home"),
                pl.col("strength_attack_away").alias("opponent_strength_attack_away"),
                pl.col("strength_defence_home").alias("opponent_strength_defence_home"),
                pl.col("strength_defence_away").alias("opponent_strength_defence_away"),
                pl.col("strength_overall_home").alias("opponent_strength_overall_home"),
                pl.col("strength_overall_away").alias("opponent_strength_overall_away"),
            ]
        ),
        how="left",
        on=["season", "round", "opponent_id"],
    )

    return players, teams, managers


def load_elements(seasons: list[str]) -> pl.LazyFrame:
    """Load per-gameweek data for all players and managers."""

    frames = []
    for season in seasons:
        path = DATA_DIR / f"fpl/{season}/elements/*.csv"
        if glob.glob(str(path)):
            frames.append(
                pl.scan_csv(
                    path,
                    try_parse_dates=True,
                    raise_if_empty=False,
                ).with_columns(pl.lit(season).alias("season"))
            )

    elements = pl.concat(frames, how="diagonal_relaxed")

    # Add the "starts" column when unavailable
    if all(season < "2022-23" for season in seasons):
        elements = elements.with_columns(pl.lit(None).cast(pl.Int64).alias("starts"))

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
            DATA_DIR / f"fpl/{season}/fixtures.csv",
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


def load_static_elements(season: str, gameweek: int) -> pl.LazyFrame:
    """Load static elements data for the given season and gameweek."""
    bootstrap_static = load_bootstrap_static(season, gameweek)
    schema_overrides = {
        "ep_next": pl.Float64,
        "ep_this": pl.Float64,
        "expected_assists": pl.Float64,
        "expected_goal_involvements": pl.Float64,
        "expected_goals": pl.Float64,
        "expected_goals_conceded": pl.Float64,
        "form": pl.Float64,
        "influence": pl.Float64,
        "creativity": pl.Float64,
        "threat": pl.Float64,
        "ict_index": pl.Float64,
        "points_per_game": pl.Float64,
        "selected_by_percent": pl.Float64,
        "value_form": pl.Float64,
        "value_season": pl.Float64,
    }
    static_elements = pl.from_dicts(
        bootstrap_static["elements"],
        schema_overrides=schema_overrides,
    )
    static_elements = static_elements.with_columns(
        pl.lit(season).alias("season"),
        pl.lit(gameweek).alias("round"),
    )
    # Add availability columns for seasons before the 2021-22 season
    if season < "2021-22":
        static_elements = static_elements.with_columns(
            pl.lit(None).cast(pl.Int64).alias("chance_of_playing_next_round"),
            pl.lit(None).cast(pl.String).alias("status"),
            pl.lit(None).cast(pl.String).alias("news"),
            pl.lit(None).cast(pl.Datetime(time_zone="UTC")).alias("news_added"),
        )
    else:
        static_elements = static_elements.with_columns(
            pl.col("news_added").str.to_datetime()
        )
    return static_elements.lazy()


def load_static_teams(season: str, gameweek: int) -> pl.LazyFrame:
    """Load static teams data for the given season and gameweek."""
    bootstrap_static = load_bootstrap_static(season, gameweek)
    static_teams = pl.from_dicts(bootstrap_static["teams"])
    static_teams = static_teams.with_columns(
        pl.lit(season).alias("season"),
        pl.lit(gameweek).alias("round"),
    )
    # Add strength columns for seasons before the 2021-22 season
    if season < "2021-22":
        static_teams = static_teams.with_columns(
            pl.lit(None).cast(pl.Int64).alias("strength"),
            pl.lit(None).cast(pl.Int64).alias("strength_attack_home"),
            pl.lit(None).cast(pl.Int64).alias("strength_attack_away"),
            pl.lit(None).cast(pl.Int64).alias("strength_defence_home"),
            pl.lit(None).cast(pl.Int64).alias("strength_defence_away"),
            pl.lit(None).cast(pl.Int64).alias("strength_overall_home"),
            pl.lit(None).cast(pl.Int64).alias("strength_overall_away"),
        )
    return static_teams.lazy()


def load_bootstrap_static(season: str, gameweek: int) -> dict:
    """Load bootstrap static data for the given season and gameweek."""
    path = DATA_DIR / f"fpl/{season}/static/{gameweek}.json.xz"
    with lzma.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def get_gameweeks(season: str) -> list[int]:
    """Scan the data directory for all gameweeks with static data."""

    path = DATA_DIR / f"fpl/{season}/static"
    if not path.exists():
        raise FileNotFoundError(f"Data directory for season {season} does not exist.")

    gameweeks = []
    for child in path.glob("*.json.xz"):
        name = child.name.split(".")[0]
        if not name.isdigit():
            continue
        gameweeks.append(int(name))

    return sorted(gameweeks)
