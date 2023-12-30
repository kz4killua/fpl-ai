"""Functions for updating local Fantasy Premier League data."""

from pathlib import Path
import json

import pandas as pd
from tqdm import tqdm

from api.fpl import get_player_data, get_fixture_data, get_bootstrap_data
from datautil.constants import LOCAL_DATA_PATH


def update_local_players(season, elements, events):
    """Update Fantasy Premier League data for players."""

    # Check the most recent gameweek
    last_updated_gameweek = int(events[events['data_checked'] == True]['id'].max())

    # Check if the local data is already up to date
    checkpoint = LOCAL_DATA_PATH / f"api/{season}/local_players_last_update.json"

    if checkpoint.exists():
        with open(checkpoint, 'r') as f:
            if last_updated_gameweek == json.load(f):
                return
    else:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)

    # Save each player's data
    for element in tqdm(elements['id'].unique(), desc="Updating local players"):
        
        # Get the player's season summary
        data = get_player_data(element)['history']

        # Create the folder to store the data
        path = LOCAL_DATA_PATH / f"api/{season}/players/{element}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save the data in CSV format
        pd.DataFrame(data).to_csv(path, index=False)

    # Keep track data updates
    with open(checkpoint, 'w') as f:
        json.dump(last_updated_gameweek, f)


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
