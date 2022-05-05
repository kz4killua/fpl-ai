import os
import subprocess
import json

import pandas as pd
import numpy as np

import api


DATE_FORMAT = r'%Y-%m-%dT%H:%M:%SZ'
ROLLING_STATISTICS = ['total_points']


def year_to_season(year):
    """Returns a string representing a season of format yyyy-yy."""
    return  str(year) + '-' + str(year + 1)[-2:]

def get_last_checked_event(events):
    """Returns the id of the last updated event."""
    return int(events[events['data_checked']==True].nlargest(1, 'id').iloc[0]['id'])

def rolling_average(column, n):
    """Calculates the average value in a series
    over the last n occurences excluding the current one."""

    # Calculate the rolling average.
    rolling_average = column.rolling(window=n, min_periods=1, closed='left').mean()
    # Fill blanks.
    rolling_average = rolling_average.fillna(0)

    return rolling_average.values

def update_vastaav_repo():
    """Updates my local clone of the github.com/vastaav/Fantasy-Premier-League repository."""
    # Get the full repo path.
    path = os.path.abspath('data/vastaav')
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
    with open("data/api/2021-22/players/last_checked.json", 'r') as f:
        n = json.load(f)
    if n == last_checked:
        return
     # Update data.   
    for id in elements['id'].unique():
        data = api.get_player_data(id)
        data = pd.DataFrame(data['history'])
        data.to_csv(f"data/api/{season}/players/{id}.csv", index=False)
    # Keep track of when last we saved player data.
    with open("data/api/2021-22/players/last_checked.json", 'w') as f:
        json.dump(last_checked, f)


def get_player_training_data(df):
    
    # Convert the kickoff times to date objects.
    df['kickoff_time'] = pd.to_datetime(df['kickoff_time'], format=DATE_FORMAT)
    # Sort the data by dates.
    df.sort_values('kickoff_time', ignore_index=True, inplace=True)

    # Compute rolling statistics.
    for column in ROLLING_STATISTICS:
        df['avg_' + column + '_5_games'] = rolling_average(df[column], 5)
        df['avg_' + column] = rolling_average(df[column], len(df))

    # Check if the player got a red card in their last match.
    df['is_suspended'] = df['red_cards'].shift(1).fillna(0).astype('int')

    return df

def get_all_training_data(season, players, teams, fixtures):

    # Create a list to hold all the data.
    data = []

    # Navigate to the data/season/players directory.
    path = f'data/vastaav/data/{season}/players'

    # Scan through every folder in that directory.
    for item in os.scandir(path):
        
        if item.is_dir():
            gw_path = os.path.join(item.path, 'gw.csv')

            # print('Processing', gw_path)

            # Read the gw.csv file into a dataframe.
            df = pd.read_csv(gw_path)

            # Get each player's data.
            player_data = get_player_training_data(df)

            # Add to the list.
            data.append(player_data)

    # Merge all the data into a single dataframe.
    df = pd.concat(data, ignore_index=True)

    # Map the home and away teams.
    df['team_h'] = df.fixture.map(fixtures.set_index('id').team_h)
    df['team_a'] = df.fixture.map(fixtures.set_index('id').team_a)

    # Map fixture difficulty ratings.
    df['team_h_difficulty'] = df.fixture.map(fixtures.set_index('id').team_h_difficulty)
    df['team_a_difficulty'] = df.fixture.map(fixtures.set_index('id').team_a_difficulty)

    # Separate home and away matches.
    home = df[df['was_home'] == True].copy()
    away = df[df['was_home'] == False].copy()
    
    # Identify teams.
    home['team'] = home['team_h']
    away['team'] = away['team_a']

    # Identify fixture difficulty ratings.
    home['team_fixture_difficulty'] = home['team_h_difficulty']
    away['team_fixture_difficulty'] = away['team_a_difficulty']
    home['opponent_fixture_difficulty'] = home['team_a_difficulty']
    away['opponent_fixture_difficulty'] = away['team_h_difficulty']
    
    # Recombine the two dataframes.
    df = pd.concat([home, away], ignore_index=True)

    # Add player position information.
    df['element_type'] = df.element.map(players.set_index('id').element_type)

    df.drop(columns=[
        'team_a_score', 'team_h_score', 'team_a', 'team_h',
        'team_h_difficulty', 'team_a_difficulty', 'value',
        'selected', 'transfers_balance', 'transfers_in', 'transfers_out',
    ], inplace=True)

    # Transform features.
    df['was_home'] = df['was_home'].astype('int64')

    # Rename columns.
    df.rename(columns={'was_home': 'is_home', 'element': 'id'}, inplace=True)

    return df


def get_player_current_data(df, next_gameweek):
    """Extracts a player's stats before a given gameweek."""

    # Filter the data to events before the given gameweek.
    df = df[df['round'] < next_gameweek].copy().reset_index()

    if len(df) == 0:
        return None

    # Convert the kickoff times to date objects.
    df['kickoff_time'] = pd.to_datetime(df['kickoff_time'], format=DATE_FORMAT)
    # Sort rows by dates.
    df.sort_values('kickoff_time', ignore_index=True, inplace=True)

    # We will store player data in a series.
    data = pd.Series(dtype='float64')

    # Get the player id.
    data.loc['id'] = df.loc[0, 'element']

    # Compute rolling statistics.
    for column in ROLLING_STATISTICS:
        data.loc['avg_' + column + '_5_games'] = df[column].iloc[-5:].mean()
        data.loc['avg_' + column] = df[column].mean()

    # Check if the player got a red card in their last match.
    data.loc['is_suspended'] = df.iloc[-1]['red_cards']

    return data

def get_all_current_data(season, elements, teams, next_gameweek):

    data = []
    # Read data for each player
    for item in os.scandir(f'data/api/{season}/players'):
        if item.path.endswith('.csv'):
            df = pd.read_csv(item.path)
            player_data = get_player_current_data(df, next_gameweek)
            if player_data is None:
                continue
            data.append(player_data)

    df = pd.DataFrame(data)
    # Map other information.
    df['element_type'] = df['id'].map(elements.set_index('id')['element_type'])
    df['team'] = df['id'].map(elements.set_index('id')['team'])

    return df


def collect_training_data():

    print('Starting...')

    seasons = ['2019-20', '2020-21']
    
    for season in seasons:

        # Load accessory files.
        players = pd.read_csv(f'data/vastaav/data/{season}/players_raw.csv')
        fixtures = pd.read_csv(f'data/vastaav/data/{season}/fixtures.csv')
        teams = pd.read_csv(f'data/vastaav/data/{season}/teams.csv')

        # Get training data.
        data = get_all_training_data(season, players, teams, fixtures)

        # Store the season.
        data['season'] = season

        # Save as a CSV file.
        data.to_csv(f'data/{season}.csv', index=False)

        print(f'Done with {season}!')

    input('Done! Press Enter to close.')


def main():
    collect_training_data()

if __name__ == '__main__':
    main()