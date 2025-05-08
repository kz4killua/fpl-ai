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
