import polars as pl

from datautil.constants import DATA_DIR
from datautil.load.fplcache import load_static_elements, load_static_teams


def load_fpl(seasons: list[str]) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Load local FPL data for the given seasons."""

    # Load local data
    elements = load_elements(seasons)
    fixtures = load_fixtures(seasons)
    static_teams = pl.concat(
        [load_static_teams(season, latest=True) for season in seasons],
        how="diagonal_relaxed",
    )
    static_elements = pl.concat(
        [load_static_elements(season, latest=True) for season in seasons],
        how="diagonal_relaxed",
    )

    # Add "element_type" and "code" to elements
    elements = elements.join(
        static_elements.select(["season", "id", "element_type", "code"]),
        how="left",
        left_on=["season", "element"],
        right_on=["season", "id"],
    )

    # Add "team" to elements
    elements = elements.join(
        fixtures.select(["id", "season", "team_h", "team_a"]),
        how="left",
        left_on=["season", "fixture"],
        right_on=["season", "id"],
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
                pl.col("id").alias("team"),
                pl.col("code").alias("team_code"),
            ]
        ),
        how="left",
        on=["season", "team"],
    )
    elements = elements.join(
        static_teams.select(
            [
                pl.col("season"),
                pl.col("id").alias("opponent_team"),
                pl.col("code").alias("opponent_team_code"),
            ]
        ),
        how="left",
        on=["season", "opponent_team"],
    )
    elements = elements.drop(["team"])

    # Split elements into players and managers
    players = elements.filter(pl.col("element_type").is_in([1, 2, 3, 4]))
    managers = elements.filter(pl.col("element_type").is_in([5]))

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

    return players, managers


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
    return pl.concat(frames, how="diagonal_relaxed")
