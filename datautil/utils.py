from collections.abc import Iterable

import polars as pl


def convert_year_to_season(year: int):
    """Converts a numeric year eg. 2023 to a string of format yyyy-yy eg. 2023-24."""
    return f"{str(year)}-{str(year + 1)[-2:]}"


def convert_year_col_to_season_col(name: str) -> pl.Expr:
    """Same as `convert_year_to_season` but for a Polars column."""
    return (
        pl.col(name).cast(pl.String)
        + "-"
        + (pl.col(name) + 1).cast(pl.String).str.slice(-2, 2)
    )


def convert_season_to_year(season: str):
    """Converts a season of format yyyy-yy eg. 2023-24 to a numeric value eg. 2023."""
    return int(season[:4])


def convert_season_col_to_year_col(name: str) -> pl.Expr:
    """Same as `convert_season_to_year` but for a Polars column."""
    return pl.col(name).str.slice(0, 4).cast(pl.Int32)


def get_seasons(current_season: str, n: int = None) -> list[str]:
    """Returns a list of seasons up to the current season."""
    min_year = 2016 if n is None else convert_season_to_year(current_season) - n + 1
    max_year = convert_season_to_year(current_season)
    return [convert_year_to_season(year) for year in range(min_year, max_year + 1)]


def get_mapper(df: pl.DataFrame, from_col: str | Iterable[str], to_col: str) -> dict:
    """Returns a dict mapping values in `from_col` to values in `to_col`."""

    if not isinstance(from_col, str | Iterable):
        raise ValueError(
            "Argument 'from_col' must be a string or an iterable of strings"
        )
    if not df.select(from_col).is_unique().all():
        raise ValueError(f"Column(s): {from_col} must be unique")

    mapper = dict()
    if isinstance(from_col, str):
        for row in df.to_dicts():
            key = row[from_col]
            mapper[key] = row[to_col]
    else:
        for row in df.to_dicts():
            key = tuple(row[col] for col in from_col)
            mapper[key] = row[to_col]

    return mapper


def calculate_implied_probabilities(home: pl.Expr, away: pl.Expr, draw: pl.Expr):
    """Convert bookmaker odds to implied probabilities, adjusted for overround."""
    implied_home = 1 / home
    implied_away = 1 / away
    implied_draw = 1 / draw
    overround = implied_home + implied_away + implied_draw

    normalized_home = implied_home / overround
    normalized_away = implied_away / overround
    normalized_draw = implied_draw / overround
    return normalized_home, normalized_away, normalized_draw


def get_teams_view(matches: pl.LazyFrame) -> pl.LazyFrame:
    """Convert per-match data to per-team data."""
    
    fixed_columns, home_columns, away_columns = [], [], []
    for column in matches.columns:
        if column.startswith("team_h_"):
            home_columns.append(column)
        elif column.startswith("team_a_"):
            away_columns.append(column)
        else:
            fixed_columns.append(column)

    home_teams = (
        matches.select(fixed_columns + home_columns)
        .rename({column: column.removeprefix("team_h_") for column in home_columns})
        .with_columns(pl.lit(1).alias("was_home"))
    )
    away_teams = (
        matches.select(fixed_columns + away_columns)
        .rename({column: column.removeprefix("team_a_") for column in away_columns})
        .with_columns(pl.lit(0).alias("was_home"))
    )

    return pl.concat([home_teams, away_teams], how="vertical")


def get_matches_view(
    teams: pl.LazyFrame, extra_fixed_columns: list[str] | None = None
) -> pl.LazyFrame:
    """Convert per-team data to per-match data."""

    fixed_columns = [
        "season",
        "round",
        "fixture_id",
        "kickoff_time",
        "was_home",
    ]
    if extra_fixed_columns:
        fixed_columns.extend(extra_fixed_columns)

    home_teams = teams.filter(pl.col("was_home") == 1).rename(
        {
            column: f"team_h_{column}"
            for column in teams.columns
            if column not in fixed_columns
        }
    )
    away_teams = teams.filter(pl.col("was_home") == 0).rename(
        {
            column: f"team_a_{column}"
            for column in teams.columns
            if column not in fixed_columns
        }
    )

    join_keys = ["season", "round", "fixture_id"]
    drop = list(set(fixed_columns) - set(join_keys))
    matches = home_teams.join(away_teams.drop(drop), on=join_keys, how="inner")
    return matches
