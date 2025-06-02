import json
import lzma

import polars as pl

from datautil.constants import DATA_DIR


def load_static_elements(
    season: str, gameweek: int | None = None, *, latest: bool = False
) -> pl.LazyFrame:
    """Load static elements data for the given season and gameweek."""
    gameweek = _resolve_gameweek(season, gameweek, latest)
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
    # Add the "news_added" column for the "2016-17" season
    if season == "2016-17":
        static_elements = static_elements.with_columns(
            pl.lit(None).cast(pl.Datetime(time_zone="UTC")).alias("news_added")
        )
    else:
        static_elements = static_elements.with_columns(
            pl.col("news_added").str.to_datetime()
        )
    return static_elements.lazy()


def load_static_teams(
    season: str, gameweek: int | None = None, *, latest: bool = False
) -> pl.LazyFrame:
    """Load static teams data for the given season and gameweek."""
    gameweek = _resolve_gameweek(season, gameweek, latest)
    bootstrap_static = load_bootstrap_static(season, gameweek)
    static_teams = pl.from_dicts(bootstrap_static["teams"])
    static_teams = static_teams.with_columns(
        pl.lit(season).alias("season"),
        pl.lit(gameweek).alias("round"),
    )
    return static_teams.lazy()


def load_bootstrap_static(
    season: str, gameweek: int | None = None, *, latest: bool = False
) -> dict:
    """Load bootstrap static data for the given season and gameweek."""
    gameweek = _resolve_gameweek(season, gameweek, latest=latest)
    path = DATA_DIR / f"fplcache/{season}/{gameweek}.json.xz"
    with lzma.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _resolve_gameweek(season: str, gameweek: int | None, latest: bool):
    if (gameweek is None and not latest) or (gameweek is not None and latest):
        raise ValueError("Either gameweek or latest must be specified, but not both.")
    if latest:
        gameweek = get_latest_gameweek(season)
    return gameweek


def get_latest_gameweek(season: str) -> int:
    """Scan the data directory for the latest gameweek of a given season."""
    gameweeks = get_gameweeks(season)
    if not gameweeks:
        raise FileNotFoundError(f"No gameweeks found for season {season}.")
    return max(gameweeks)


def get_gameweeks(season: str) -> list[int]:
    """Scan the data directory for all gameweeks of a given season."""

    path = DATA_DIR / f"fplcache/{season}"
    if not path.exists():
        raise FileNotFoundError(f"Data directory for season {season} does not exist.")
    
    gameweeks = []
    for child in path.glob("*.json.xz"):
        gameweek = int(child.name.split(".")[0])
        gameweeks.append(gameweek)
        
    return sorted(gameweeks)