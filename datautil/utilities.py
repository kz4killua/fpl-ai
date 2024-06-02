import pandas as pd


def convert_year_to_season(year: int):
    """
    Convert a numeric season starting year eg. 2023 to a string of format yyyy-yy eg. 2023-24
    """
    return f'{str(year)}-{str(year + 1)[-2:]}'

def convert_season_to_year(season: str):
    """
    Convert a season of format yyyy-yy eg. 2023-24 to a numeric value eg. 2023
    """
    return int(season[:4])

def get_next_gameweek(events: pd.DataFrame):
    return events[events['is_next'] == True].iloc[0]['id']

def get_current_season(events: pd.DataFrame):
    year = events['deadline_time'].min().year
    return convert_year_to_season(year)

def get_previous_seasons(current_season: str):
    """
    Returns a list of all seasons (including the current one) since the 2016-17 season.
    """
    current_year = convert_season_to_year(current_season)
    return [
        convert_year_to_season(year) for year in range(2016, current_year + 1)
    ]