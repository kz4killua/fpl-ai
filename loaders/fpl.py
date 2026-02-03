import glob
import json
import lzma

import polars as pl

from game.rules import DEF, FWD, GKP, MID, MNG
from loaders.constants import DATA_DIR
from loaders.upcoming import (
    get_upcoming_elements,
    get_upcoming_fixtures,
    get_upcoming_static_elements,
    get_upcoming_static_teams,
    remove_future,
)
from loaders.utils import get_teams_view


def load_fpl(
    seasons: list[int],
    current_season: int | None = None,
    upcoming_gameweeks: list[int] | None = None,
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

        # Create blank records for upcoming gameweeks
        upcoming_fixtures = get_upcoming_fixtures(
            fixtures, current_season, upcoming_gameweeks
        )
        upcoming_elements = get_upcoming_elements(
            current_season, upcoming_gameweeks, upcoming_fixtures, static_elements
        )
        upcoming_static_teams = get_upcoming_static_teams(
            current_season, upcoming_gameweeks, static_teams
        )
        upcoming_static_elements = get_upcoming_static_elements(
            current_season, upcoming_gameweeks, static_elements
        )

        # Remove known records for future gameweeks
        fixtures = remove_future(fixtures, current_season, min(upcoming_gameweeks))
        elements = remove_future(elements, current_season, min(upcoming_gameweeks))
        static_elements = remove_future(
            static_elements, current_season, min(upcoming_gameweeks)
        )
        static_teams = remove_future(
            static_teams, current_season, min(upcoming_gameweeks)
        )

        # Add records for upcoming gameweeks
        fixtures = pl.concat([fixtures, upcoming_fixtures], how="diagonal_relaxed")
        elements = pl.concat([elements, upcoming_elements], how="diagonal_relaxed")
        static_elements = pl.concat(
            [static_elements, upcoming_static_elements], how="diagonal_relaxed"
        )
        static_teams = pl.concat(
            [static_teams, upcoming_static_teams], how="diagonal_relaxed"
        )

    # Add "element_type" and "code" to elements
    elements = elements.join(
        static_elements.select(
            [
                pl.col("season"),
                pl.col("gameweek"),
                pl.col("id").alias("element"),
                pl.col("element_type"),
                pl.col("code"),
            ]
        ),
        how="left",
        on=["season", "gameweek", "element"],
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
    for column in ["team", "opponent_team"]:
        elements = elements.join(
            static_teams.select(
                [
                    pl.col("season"),
                    pl.col("gameweek"),
                    pl.col("id").alias(column),
                    pl.col("code").alias(f"{column}_code"),
                ]
            ),
            how="left",
            on=["season", "gameweek", column],
        )

    # Compute defensive contributions for seasons before the 2024-2025 season
    if any(season < 2025 for season in seasons):
        elements = elements.with_columns(
            pl.when(pl.col("defensive_contribution").is_not_null())
            .then(pl.col("defensive_contribution"))
            .when(pl.col("element_type") == GKP)
            .then(pl.lit(0))
            .when(pl.col("element_type") == DEF)
            .then(pl.col("clearances_blocks_interceptions") + pl.col("tackles"))
            .when(pl.col("element_type").is_in([MID, FWD]))
            .then(
                pl.col("clearances_blocks_interceptions")
                + pl.col("tackles")
                + pl.col("recoveries")
            )
            .otherwise(pl.col("defensive_contribution"))
            .alias("defensive_contribution")
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
        # These columns are only available from the 2024-2025 season
        strict=False,
    )

    # Map availability and set piece information to players
    players = players.join(
        static_elements.select(
            [
                "season",
                "gameweek",
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
        on=["season", "gameweek", "code"],
        how="left",
    )

    # Load team information
    matches = fixtures.select(
        pl.col("season"),
        pl.col("id").alias("fixture_id"),
        pl.col("gameweek"),
        pl.col("kickoff_time"),
        pl.col("team_h").alias("team_h_id"),
        pl.col("team_a").alias("team_a_id"),
        pl.col("team_h_score").alias("team_h_goals_scored"),
        pl.col("team_a_score").alias("team_a_goals_scored"),
    )
    matches = matches.with_columns(
        pl.col("team_h_goals_scored").alias("team_a_goals_conceded"),
        pl.col("team_a_goals_scored").alias("team_h_goals_conceded"),
    )
    teams = get_teams_view(matches)

    # Add team codes
    teams = teams.join(
        static_teams.select(
            [
                pl.col("season"),
                pl.col("gameweek"),
                pl.col("id"),
                pl.col("code"),
            ]
        ),
        how="left",
        on=["season", "gameweek", "id"],
    )

    # Add team strengths
    teams = teams.join(
        static_teams.select(
            [
                pl.col("season"),
                pl.col("gameweek"),
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
        on=["season", "gameweek", "id"],
    )

    return players, teams, managers


def load_elements(seasons: list[int]) -> pl.LazyFrame:
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

    elements: pl.LazyFrame = pl.concat(frames, how="diagonal_relaxed")

    # Add the "starts" column when unavailable
    if all(season < 2022 for season in seasons):
        elements = elements.with_columns(pl.lit(None).cast(pl.Int64).alias("starts"))

    # Add columns for defensive contribution calculations when unavailable
    if all(2018 < season < 2025 for season in seasons):
        elements = elements.with_columns(
            pl.lit(None).cast(pl.Int64).alias("clearances_blocks_interceptions"),
            pl.lit(None).cast(pl.Int64).alias("tackles"),
            pl.lit(None).cast(pl.Int64).alias("recoveries"),
        )

    if all(season < 2025 for season in seasons):
        elements = elements.with_columns(
            pl.lit(None).cast(pl.Int64).alias("defensive_contribution")
        )

    # Remove discontinued features
    elements = elements.drop(
        [
            "attempted_passes",
            "big_chances_created",
            "big_chances_missed",
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
            "tackled",
            "target_missed",
            "winning_goals",
        ],
        strict=False,
    )

    # Rename columns for consistency
    elements = elements.rename({"round": "gameweek"})

    return elements


def load_fixtures(seasons: list[int]) -> pl.LazyFrame:
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
            pl.col("season"),
            pl.col("id"),
            pl.col("event").alias("gameweek"),
            pl.col("kickoff_time"),
            pl.col("team_h"),
            pl.col("team_a"),
            pl.col("team_h_score"),
            pl.col("team_a_score"),
        ]
    )

    return fixtures


def load_static_elements(season: int, gameweek: int) -> pl.LazyFrame:
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
    static_elements = pl.LazyFrame(
        bootstrap_static["elements"],
        schema_overrides=schema_overrides,
    )
    static_elements = static_elements.with_columns(
        pl.lit(season).alias("season"),
        pl.lit(gameweek).alias("gameweek"),
    )
    # Add availability columns for seasons before the 2021/2022 season
    if season < 2021:
        static_elements = static_elements.with_columns(
            pl.lit(None).cast(pl.Int64).alias("chance_of_playing_next_round"),
            pl.lit(None).cast(pl.String).alias("status"),
            pl.lit(None).cast(pl.String).alias("news"),
            pl.lit(None).cast(pl.Datetime(time_zone="UTC")).alias("news_added"),
        )
    else:
        static_elements = static_elements.with_columns(
            pl.col("news_added").str.to_datetime(time_zone="UTC")
        )
    return static_elements


def load_static_teams(season: int, gameweek: int) -> pl.LazyFrame:
    """Load static teams data for the given season and gameweek."""
    bootstrap_static = load_bootstrap_static(season, gameweek)
    static_teams = pl.LazyFrame(bootstrap_static["teams"])
    static_teams = static_teams.with_columns(
        pl.lit(season).alias("season"),
        pl.lit(gameweek).alias("gameweek"),
    )
    # Add strength columns for seasons before the 2021/2022 season
    if season < 2021:
        static_teams = static_teams.with_columns(
            pl.lit(None).cast(pl.Int64).alias("strength"),
            pl.lit(None).cast(pl.Int64).alias("strength_attack_home"),
            pl.lit(None).cast(pl.Int64).alias("strength_attack_away"),
            pl.lit(None).cast(pl.Int64).alias("strength_defence_home"),
            pl.lit(None).cast(pl.Int64).alias("strength_defence_away"),
            pl.lit(None).cast(pl.Int64).alias("strength_overall_home"),
            pl.lit(None).cast(pl.Int64).alias("strength_overall_away"),
        )
    return static_teams


def load_bootstrap_static(season: int, gameweek: int) -> dict:
    """Load bootstrap static data for the given season and gameweek."""
    path = DATA_DIR / f"fpl/{season}/static/{gameweek}.json.xz"
    with lzma.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def get_gameweeks(season: int) -> list[int]:
    """Scan the data directory for all available gameweeks."""

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
