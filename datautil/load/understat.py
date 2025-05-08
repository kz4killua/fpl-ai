import polars as pl

from datautil.constants import DATA_DIR
from datautil.utils import convert_season_to_year


def load_player_matches() -> pl.LazyFrame:
    """Load player match data."""
    return pl.scan_csv(
        DATA_DIR / "understat/player/matches/*.csv",
        try_parse_dates=True,
        raise_if_empty=False,
    )


def load_teams(seasons: list[str]) -> pl.LazyFrame:
    """Load team data."""
    years = map(convert_season_to_year, seasons)
    frames = [
        pl.scan_csv(
            DATA_DIR / f"understat/season/{year}/teams/*.csv",
            try_parse_dates=True,
            raise_if_empty=False,
        )
        for year in years
    ]
    return pl.concat(frames, how="diagonal")


def load_dates(seasons: list[str]) -> pl.LazyFrame:
    """Load date data."""
    years = map(convert_season_to_year, seasons)
    frames = [
        pl.scan_csv(
            DATA_DIR / f"understat/season/{year}/dates.csv",
            try_parse_dates=True,
            raise_if_empty=False,
        )
        for year in years
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
    years = map(convert_season_to_year, seasons)
    frames = [
        pl.scan_csv(
            DATA_DIR / f"understat/season/{year}/fixture_ids.csv",
            try_parse_dates=True,
            raise_if_empty=False,
        )
        for year in years
    ]
    return pl.concat(frames, how="diagonal")
