def convert_year_to_season(year: int):
    """Converts a numeric year eg. 2023 to a string of format yyyy-yy eg. 2023-24."""
    return f"{str(year)}-{str(year + 1)[-2:]}"


def convert_season_to_year(season: str):
    """Converts a season of format yyyy-yy eg. 2023-24 to a numeric value eg. 2023."""
    return int(season[:4])
