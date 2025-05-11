import json
import lzma

import polars as pl

from datautil.constants import DATA_DIR


def load_static_elements(
    season: str, gameweek: int | None = None, *, latest: bool = False
) -> pl.LazyFrame:
    """Load static elements data for the given season and gameweek."""
    bootstrap_static = load_bootstrap_static(season, gameweek, latest=latest)
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
        pl.lit(season).alias("season"), pl.col("news_added").str.to_datetime()
    )
    return static_elements.lazy()


def load_static_teams(
    season: str, gameweek: int | None = None, *, latest: bool = False
) -> pl.LazyFrame:
    """Load static teams data for the given season and gameweek."""
    bootstrap_static = load_bootstrap_static(season, gameweek, latest=latest)
    static_teams = pl.from_dicts(bootstrap_static["teams"])
    static_teams = static_teams.with_columns(
        pl.lit(season).alias("season"),
    )
    return static_teams.lazy()


def load_bootstrap_static(
    season: str, gameweek: int | None = None, *, latest: bool = False
) -> dict:
    """Load bootstrap static data for the given season and gameweek."""

    if (gameweek is None and not latest) or (gameweek is not None and latest):
        raise ValueError("Either gameweek or latest must be specified, but not both.")

    if latest:
        gameweek = get_latest_gameweek(season)

    path = DATA_DIR / f"fplcache/{season}/{gameweek}.json.xz"
    with lzma.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def get_latest_gameweek(season: str) -> int:
    """Scan the data directory for the latest gameweek of a given season."""

    path = DATA_DIR / f"fplcache/{season}"
    if not path.exists():
        raise FileNotFoundError(f"Data directory for season {season} does not exist.")

    latest = 0
    for child in path.glob("*.json.xz"):
        gameweek = int(child.name.split(".")[0])
        if gameweek > latest:
            latest = gameweek
    if latest == 0:
        raise FileNotFoundError(f"No gameweek data found for season {season}.")

    return latest
