"""Functions for updating local Fantasy Premier League data."""

import pandas as pd
from tqdm import tqdm

from api.fpl import get_player_data, get_fixture_data
from datautil.constants import LOCAL_DATA_PATH


def update_local_players(season, elements):
    """Update Fantasy Premier League data for each player."""
    for element in tqdm(elements['id'].unique(), desc="Updating local players"):        
        data = get_player_data(element)['history']
        path = LOCAL_DATA_PATH / f"api/{season}/players/{element}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(data).to_csv(path, index=False)


def update_local_teams(season, teams):
    """Update Fantasy Premier League data for teams."""
    path = LOCAL_DATA_PATH / f"api/{season}/teams.csv"
    teams.to_csv(path)


def update_local_elements(season, elements):
    """Update elements (bootstrap) data for an FPL season."""
    path = LOCAL_DATA_PATH / f"api/{season}/elements.csv"
    elements.to_csv(path)


def update_local_fixtures(season):
    """Update fixture data for the current FPL season."""
    path = LOCAL_DATA_PATH / f"api/{season}/fixtures.csv"
    pd.DataFrame(get_fixture_data()).to_csv(path)
