"""Functions for updating local Fantasy Premier League data."""

import lzma
import os
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import git

from api.fpl import get_player_data, get_fixture_data
from datautil.constants import LOCAL_DATA_PATH
from datautil.utilities import convert_season_to_year


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


def update_bootstrap_data(season, start_month=8, end_month=5):
    """
    Find and save bootstrap data for each gameweek.
    """

    start_year = convert_season_to_year(season)
    end_year = start_year + 1
    cache = dict()

    # Update the fplcache repository
    repository = git.Repo(LOCAL_DATA_PATH / "fplcache")
    repository.remotes.origin.pull()

    # Populate the `cache` with the bootstrap data for each gameweek
    for year in (start_year, end_year):

        if year == start_year:
            months = range(start_month, 12 + 1)
        else:
            months = range(1, end_month + 1)

        for month in months:
            for day in range(1, 32):
                
                # Skip any invalid days eg. 30th February
                path = Path(LOCAL_DATA_PATH / f'fplcache/cache/{year}/{month}/{day}')
                if not path.exists():
                    continue

                for item in sorted(os.scandir(path), key=lambda item: item.name):

                    with lzma.open(item.path) as f:
                        bootstrap = json.load(f)

                    # Store the latest bootstrap data for each gameweek
                    events = pd.DataFrame(bootstrap['events'])
                    if len(events[events['is_current']]) == 0:
                        if events[events['is_next']]['id'].iloc[0] == 1:
                            cache[0] = bootstrap
                    else:
                        current_event = events[events['is_current']].iloc[0]
                        if current_event['data_checked']:
                            cache[current_event['id']] = bootstrap

    # Serialize the bootstrap objects
    for key, value in cache.items():
        with open(LOCAL_DATA_PATH / f"api/{season}/bootstrap/after_gameweek_{key}.json", 'w') as f:
            json.dump(value, f)