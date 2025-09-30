import json
import lzma
import warnings
from datetime import datetime

import polars as pl

from datautil.constants import DATA_DIR
from datautil.fpl import get_gameweeks


def load_theoddsapi(
    seasons: list[int],
    current_season: int,
    next_gameweek: int,
    cutoff_time: datetime | None = None,
):
    """Load odds data from the-odds-api.com."""

    # Load odds data for all season and gameweek combinations
    unique_matches = dict()
    for season in sorted(season for season in seasons if season >= 2021):
        for gameweek in sorted(get_gameweeks(season)):
            if season == current_season and gameweek > next_gameweek:
                break
            # Important: Sorted order prevents overwriting future data with past data
            unique_matches.update(
                _load_theoddsapi(season, gameweek, cutoff_time=cutoff_time)
            )

    # Construct a Polars DataFrame from the unique matches
    data = {
        "season": [],
        "team_h": [],
        "team_a": [],
        "commence_time": [],
        "bookmakers": [],
    }
    for match_key, match_data in unique_matches.items():
        season, team_h, team_a = match_key
        data["season"].append(season)
        data["team_h"].append(team_h)
        data["team_a"].append(team_a)
        data["commence_time"].append(
            datetime.fromisoformat(match_data["commence_time"])
        )
        data["bookmakers"].append(match_data["bookmakers"])

    df = pl.DataFrame(data)

    # Add team codes
    team_ids = pl.read_csv(DATA_DIR / "theoddsapi/team_ids.csv")
    for column in ["team_h", "team_a"]:
        df = df.join(
            team_ids.select(
                pl.col("theoddsapi_name").alias(column),
                pl.col("fpl_code").alias(f"{column}_fpl_code"),
            ),
            on=column,
            how="left",
        )

    return df.lazy()


def _load_theoddsapi(
    season: int, gameweek: int, cutoff_time: datetime | None = None
) -> pl.LazyFrame:
    """Load odds data from the-odds-api.com for a given season and gameweek."""

    path = DATA_DIR / f"theoddsapi/{season}/{gameweek}.json.xz"
    with lzma.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)

    # Filter out any bookmaker data updated after the cutoff time
    # This should not happen if data is being collected correctly
    if cutoff_time:
        for match in data:
            bookmakers = []
            for bookmaker in match["bookmakers"]:
                last_update = datetime.fromisoformat(bookmaker["last_update"])
                if last_update <= cutoff_time:
                    bookmakers.append(bookmaker)
                else:
                    warnings.warn(
                        "Ignoring bookmaker data updated after cutoff time. ",
                        stacklevel=2,
                    )
            match["bookmakers"] = bookmakers

    # Remove duplicates by picking the occurence of each match with the most bookmakers
    unique_matches = dict()
    for match in data:
        match_key = (season, match["home_team"], match["away_team"])
        if match_key not in unique_matches:
            unique_matches[match_key] = match
        else:
            warnings.warn(
                f"Found multiple occurences for {match['home_team']} vs "
                f"{match['away_team']} in the {season} season, gameweek {gameweek}. "
                f"Keeping the one with the most bookmakers...",
                stacklevel=2,
            )
            unique_matches[match_key] = max(
                match, unique_matches[match_key], key=lambda x: len(x["bookmakers"])
            )

    return unique_matches
