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

    # Check input types
    if not isinstance(from_col, str | Iterable):
        raise ValueError(
            "Argument 'from_col' must be a string or an iterable of strings"
        )
    # Ensure that mappings are unique
    if not df.select(from_col).is_unique().all():
        raise ValueError(f"Column(s): {from_col} must be unique")

    mapper = dict()
    if isinstance(from_col, str):
        for row in df.to_dicts():
            key = row[from_col]
            mapper[row[from_col]] = row[to_col]
    else:
        for row in df.to_dicts():
            key = tuple(row[col] for col in from_col)
            mapper[key] = row[to_col]

    return mapper
