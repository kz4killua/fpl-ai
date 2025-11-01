from collections.abc import Iterable

import polars as pl


def get_seasons(current_season: int, n: int = None) -> list[str]:
    """Returns a list of seasons up to the current season."""
    min_season = 2016 if n is None else (current_season - n + 1)
    max_season = current_season
    return [season for season in range(min_season, max_season + 1)]


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


def get_teams_view(matches: pl.LazyFrame) -> pl.LazyFrame:
    """Convert per-match data to per-team data."""

    fixed_columns, home_columns, away_columns = [], [], []
    for column in get_columns(matches):
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
        "gameweek",
        "fixture_id",
        "kickoff_time",
        "was_home",
    ]
    if extra_fixed_columns:
        fixed_columns.extend(extra_fixed_columns)

    home_teams = teams.filter(pl.col("was_home") == 1).rename(
        {
            column: f"team_h_{column}"
            for column in get_columns(teams)
            if column not in fixed_columns
        }
    )
    away_teams = teams.filter(pl.col("was_home") == 0).rename(
        {
            column: f"team_a_{column}"
            for column in get_columns(teams)
            if column not in fixed_columns
        }
    )

    join_keys = ["season", "gameweek", "fixture_id"]
    drop = list(set(fixed_columns) - set(join_keys))
    matches = home_teams.join(away_teams.drop(drop), on=join_keys, how="inner")
    return matches


def get_columns(df: pl.LazyFrame | pl.DataFrame) -> list[str]:
    """Get the list of columns in a Polars DataFrame or LazyFrame."""
    if isinstance(df, pl.LazyFrame):
        return df.collect_schema().names()
    elif isinstance(df, pl.DataFrame):
        return df.columns
    else:
        raise ValueError("Argument 'df' must be a Polars DataFrame or LazyFrame")


def force_dataframe(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    """Ensure the input is a Polars DataFrame."""
    if isinstance(df, pl.LazyFrame):
        return df.collect()
    return df


def force_lazyframe(df: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    """Ensure the input is a Polars LazyFrame."""
    if isinstance(df, pl.DataFrame):
        return df.lazy()
    return df
