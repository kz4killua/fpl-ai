import json
import lzma

import polars as pl

from datautil.constants import DATA_DIR
from datautil.utils import calculate_implied_probabilities, convert_season_to_year


def load_theoddsapi(season: str, gameweek: int) -> pl.LazyFrame:
    """Load upcoming odds data from the-odds-api.com for a given season and gameweek."""
    path = DATA_DIR / f"theoddsapi/{convert_season_to_year(season)}/{gameweek}.json.xz"
    with lzma.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)

    # Remove duplicates by picking the occurence of each match with the most bookmakers
    unique_matches = dict()
    for match in data:
        match_key = (match["home_team"], match["away_team"])
        if match_key not in unique_matches:
            unique_matches[match_key] = match
        else:
            unique_matches[match_key] = max(
                match, unique_matches[match_key], key=lambda x: len(x["bookmakers"])
            )

    # Keep a list of bookmakers for calculating implied probabilities later
    bookmaker_keys = set()

    rows = []
    for match in unique_matches.values():
        home_team = match["home_team"]
        away_team = match["away_team"]
        odds = {
            "home": home_team,
            "away": away_team,
        }

        for bookmaker in match["bookmakers"]:
            bookmaker_keys.add(bookmaker["key"])

            for market in bookmaker["markets"]:
                # We only care about head-to-head (h2h) markets for now
                if market["key"] == "h2h":
                    for outcome in market["outcomes"]:
                        if outcome["name"] == home_team:
                            odds[f"{bookmaker['key']}_home"] = outcome["price"]
                        elif outcome["name"] == away_team:
                            odds[f"{bookmaker['key']}_away"] = outcome["price"]
                        elif outcome["name"] == "Draw":
                            odds[f"{bookmaker['key']}_draw"] = outcome["price"]

        rows.append(odds)

    # Declare schema (in case of missing data e.g. Gameweek 1 for 2022-23)
    schema = {
        "home": pl.String,
        "away": pl.String,
    }
    df = pl.DataFrame(rows, schema=schema).with_columns(
        pl.lit(season).alias("season"),
        pl.lit(gameweek).alias("round"),
    )

    # Convert odds to implied probabilities
    for bookmaker_key in bookmaker_keys:
        implied_home, implied_away, implied_draw = calculate_implied_probabilities(
            pl.col(f"{bookmaker_key}_home"),
            pl.col(f"{bookmaker_key}_away"),
            pl.col(f"{bookmaker_key}_draw"),
        )
        df = df.with_columns(
            implied_home.alias(f"{bookmaker_key}_home_implied"),
            implied_away.alias(f"{bookmaker_key}_away_implied"),
            implied_draw.alias(f"{bookmaker_key}_draw_implied"),
        )

    # Add FPL codes for home and away teams
    team_ids = pl.read_csv(DATA_DIR / "theoddsapi/team_ids.csv")
    for column in ["home", "away"]:
        df = df.join(
            team_ids.select(
                pl.col("theoddsapi_name").alias(column),
                pl.col("fpl_code").alias(f"{column}_fpl_code"),
            ),
            on=column,
            how="left",
        )

    return df.lazy()
