"""Functions for updating local understat.com data."""

import pandas as pd
from tqdm import tqdm

from api.understat import get_league_players_data, get_player_matches_data, get_league_teams_data, get_league_dates_data
from .utilities import season_to_year
from datautil.constants import LOCAL_DATA_PATH



def save_understat_player_matches_data(season):
    """Saves understat data for all players who played in a season"""

    # Get the understat IDs of all EPL players
    players_data = get_league_players_data('EPL', season_to_year(season))
    understat_ids = pd.DataFrame(players_data)['id'].astype('int').values

    # Create the folder to save the data
    folder = LOCAL_DATA_PATH / f"understat/player/matches"
    folder.mkdir(parents=True, exist_ok=True)

    for id in tqdm(understat_ids, desc="Updating understat players"):

        # Fetch match data for each player
        player_matches_data = get_player_matches_data(id)
        player_matches_data = pd.DataFrame(player_matches_data)

        # Save the data to a CSV
        player_matches_data.to_csv(folder / f'{id}.csv', index=False)


def save_understat_league_teams_data(season):

    # Fetch understat team data
    teams_data = get_league_teams_data('EPL', season_to_year(season))

    # Create the folder to save the data
    folder = LOCAL_DATA_PATH / f"understat/season/{season_to_year(season)}/teams"
    folder.mkdir(parents=True, exist_ok=True)

    for team in teams_data.values():
        
        # Create a table of each team's history
        history = pd.DataFrame(team['history'])
        history['id'] = team['id']
        history['title'] = team['title']

        # Save the data to a CSV
        history.to_csv(folder / f"{team['id']}.csv", index=False)


def save_understat_league_dates_data(season):

    # Fetch understat dates data
    dates = get_league_dates_data('EPL', season_to_year(season))

    # Flatten the data
    for date in dates:
        date['h'] = date['h']['id']
        date['a'] = date['a']['id']

    # Convert to a dataframe
    dates = pd.DataFrame(dates)

    # Create the folder to save the data
    path = LOCAL_DATA_PATH / f'understat/season/{season_to_year(season)}/dates.csv'
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save the data to a CSV
    dates.to_csv(path, index=False)