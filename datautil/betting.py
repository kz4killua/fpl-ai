from datetime import datetime

import polars as pl

from datautil.footballdata import load_footballdata
from datautil.theoddsapi import load_theoddsapi


def load_market_probabilities(
    seasons: list[str],
    cutoff_time: datetime,
    current_season: str | None = None,
    next_gameweek: int | None = None,
):
    """Unified data loader for historical and upcoming market odds."""

    historical_odds = load_footballdata(seasons, cutoff_time)
    historical_bookmakers = [
        [
            "PSH_implied",
            "PSA_implied",
            "PSD_implied",
        ],
    ]
    historical_odds = average_implied_probabilities(
        historical_odds, historical_bookmakers
    )
    df = historical_odds.select(
        pl.col("season"),
        pl.col("HomeTeam_fpl_code").alias("home_team_code"),
        pl.col("AwayTeam_fpl_code").alias("away_team_code"),
        pl.col("home_market_probability"),
        pl.col("away_market_probability"),
        pl.col("draw_market_probability"),
    )

    if (current_season is not None) and (next_gameweek is not None):
        upcoming_odds = load_theoddsapi(current_season, next_gameweek)
        upcoming_bookmakers = [
            [
                "betfair_ex_uk_home_implied",
                "betfair_ex_uk_away_implied",
                "betfair_ex_uk_draw_implied",
            ],
            [
                "smarkets_home_implied",
                "smarkets_away_implied",
                "smarkets_draw_implied",
            ],
            [
                "matchbook_home_implied",
                "matchbook_away_implied",
                "matchbook_draw_implied",
            ],
        ]
        upcoming_odds = average_implied_probabilities(
            upcoming_odds, upcoming_bookmakers
        )
        upcoming_odds = upcoming_odds.select(
            pl.col("season"),
            pl.col("home_fpl_code").alias("home_team_code"),
            pl.col("away_fpl_code").alias("away_team_code"),
            pl.col("home_market_probability"),
            pl.col("away_market_probability"),
            pl.col("draw_market_probability"),
        )
        df = pl.concat([df, upcoming_odds], how="vertical")

    return df


def average_implied_probabilities(df: pl.LazyFrame, bookmakers: list[tuple]):
    """Average the implied probabilities from different bookmakers."""

    schema = set(df.collect_schema().names())
    for bookmaker in bookmakers:
        for column in bookmaker:
            if column not in schema:
                df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(column))

    return df.with_columns(
        pl.mean_horizontal(bookmaker[0] for bookmaker in bookmakers).alias(
            "home_market_probability"
        ),
        pl.mean_horizontal(bookmaker[1] for bookmaker in bookmakers).alias(
            "away_market_probability"
        ),
        pl.mean_horizontal(bookmaker[2] for bookmaker in bookmakers).alias(
            "draw_market_probability"
        ),
    )
