import os
import subprocess
import json

import pandas as pd

import api


DATE_FORMAT = r'%Y-%m-%dT%H:%M:%SZ'
ROLLING_STATISTICS = (
    'total_points', 'influence', 'creativity', 'threat', 
    'ict_index', 'was_home', 'goals_scored', 'goals_conceded', 
    'assists', 'clean_sheets', 'bonus', 'bps', 'minutes')
ROLLING_WINDOWS = (3, 40)


def year_to_season(year):
    """Returns a string representing a season of format yyyy-yy."""
    return  str(year) + '-' + str(year + 1)[-2:]


def season_to_year(season):
    """Returns the year season 'ssss-ss' started."""
    return int(season[:4])


def get_last_checked_event(events):
    """Returns the id of the last updated event."""
    return int(events[events['data_checked']==True].nlargest(1, 'id').iloc[0]['id'])


def get_current_season(fixtures):
    """Returns the current season."""
    year = int(fixtures.dropna().iloc[0]['kickoff_time'].split('-')[0])
    return year_to_season(year)


def rolling_average(column, n):
    """Calculates the average value in a series
    over the last n occurences excluding the current one."""

    # Calculate the rolling average.
    rolling_average = column.rolling(window=n, min_periods=1, closed='left').mean()
    # Fill blanks.
    rolling_average = rolling_average.fillna(0)

    return rolling_average.values


def update_vaastav_repo():
    """Updates my local clone of the github.com/vaastav/Fantasy-Premier-League repository."""
    # Get the full repo path.
    path = os.path.abspath('data/vaastav')
    # This is the command we will run.
    command = f'cd {path} && git pull origin master'
    # Capture output.
    output = subprocess.run(command, capture_output=True, shell=True)
    # Print output.
    print(output.stdout.decode())


def update_players_data(season, elements, events):
    """Saves and stores most recent data for all players."""
    # Check whether the data is already up to date.
    last_checked = get_last_checked_event(events)
    with open(f"data/api/{season}/players/last_checked.json", 'r') as f:
        n = json.load(f)
    if n == last_checked:
        return
     # Update data.   
    for id in elements['id'].unique():
        data = api.get_player_data(id)
        data = pd.DataFrame(data['history'])
        data.to_csv(f"data/api/{season}/players/{id}.csv", index=False)
    # Keep track of when last we saved player data.
    with open(f"data/api/{season}/players/last_checked.json", 'w') as f:
        json.dump(last_checked, f)


def get_current_data(season, elements, next_gameweek):
    """Fetches current player data for making predictions."""

    elements = elements.set_index('id', drop=False)

    # Load player data for the previous season
    previous_season = year_to_season(season_to_year(season) - 1)
    previous = pd.read_csv(f'data/api/merged/players/{previous_season}.csv')
    previous['season'] = previous_season

    # Load player data for the current season
    current = list()
    for item in os.scandir(f'data/api/{season}/players'):
        if item.path.endswith('.csv'):
            try:
                player = pd.read_csv(item.path)
            except pd.errors.EmptyDataError:
                continue
            player_id = int(item.name[:-4])
            if player_id in elements['id']:
                player['code'] = elements.loc[player_id, 'code']
                current.append(player)
    current = pd.concat(current)
    current['season'] = season
    
    # Filter out unnecessary records
    current = current[current['round'] < next_gameweek]

    # Combine both dataframes
    players = pd.concat([previous, current])

    # Sort player records by kickoff times
    players['kickoff_time'] = pd.to_datetime(players['kickoff_time'], format=DATE_FORMAT)
    players.sort_values('kickoff_time', ignore_index=True, inplace=True)

    data = list()
    for index, code, id in elements[['code', 'id']].itertuples():
        player = dict()
        # Filter out each player's records
        records = players[players['code'] == code]
        # Compute rolling stats for each player
        for feature in ROLLING_STATISTICS:
            for window in ROLLING_WINDOWS:
                player[f'avg_{feature}_{window}_games'] = records[feature].iloc[-window:].mean() if len(records) else 0
        # Add other player stats
        player['id'] = id
        player['team'] = elements.loc[id, 'team']
        player['is_gkp'] = 1 if elements.loc[id, 'element_type'] == 1 else 0
        player['is_def'] = 1 if elements.loc[id, 'element_type'] == 2 else 0
        player['is_mid'] = 1 if elements.loc[id, 'element_type'] == 3 else 0
        player['is_fwd'] = 1 if elements.loc[id, 'element_type'] == 4 else 0
        # Add to the list
        data.append(player)

    data = pd.DataFrame(data)

    return data