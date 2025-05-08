import polars as pl

from datautil.constants import DATA_DIR
from datautil.utils import convert_season_to_year


def load_player_matches(seasons: list[str]) -> pl.LazyFrame:
    """Load player match data."""
    seasons = list(map(convert_season_to_year, seasons))
    return (
        pl.scan_csv(
            DATA_DIR / "understat/player/matches/*.csv",
            try_parse_dates=True,
            raise_if_empty=False,
            include_file_paths="file_path",
        )
        .with_columns(
            pl.col("file_path")
            .str.extract(r"(\d+)\.csv")
            .cast(pl.Int32)
            .alias("player_id"),
        )
        .filter(pl.col("season").is_in(seasons))
    )


def load_teams(seasons: list[str]) -> pl.LazyFrame:
    """Load team data."""
    seasons = list(map(convert_season_to_year, seasons))
    frames = [
        pl.scan_csv(
            DATA_DIR / f"understat/season/{season}/teams/*.csv",
            try_parse_dates=True,
            raise_if_empty=False,
        ).with_columns(
            pl.lit(season).alias("season"),
        )
        for season in seasons
    ]
    return pl.concat(frames, how="diagonal")


def load_dates(seasons: list[str]) -> pl.LazyFrame:
    """Load date data."""
    seasons = list(map(convert_season_to_year, seasons))
    frames = [
        pl.scan_csv(
            DATA_DIR / f"understat/season/{season}/dates.csv",
            try_parse_dates=True,
            raise_if_empty=False,
        ).with_columns(
            pl.lit(season).alias("season"),
        )
        for season in seasons
    ]
    return pl.concat(frames, how="diagonal")


def load_player_ids() -> pl.LazyFrame:
    """Load FPL-to-understat player ID mappings."""
    return pl.scan_csv(
        DATA_DIR / "understat/player/player_ids.csv",
        try_parse_dates=True,
    )


def load_fixture_ids(seasons: list[str]) -> pl.LazyFrame:
    """Load FPL-to-understat fixture ID mappings."""
    seasons = list(map(convert_season_to_year, seasons))
    frames = [
        pl.scan_csv(
            DATA_DIR / f"understat/season/{season}/fixture_ids.csv",
            try_parse_dates=True,
            raise_if_empty=False,
        ).with_columns(
            pl.lit(season).alias("season"),
        )
        for season in seasons
    ]
    return pl.concat(frames, how="diagonal")
