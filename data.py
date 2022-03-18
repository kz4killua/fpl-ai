import os
import subprocess

import pandas as pd
import numpy as np


DATE_FORMAT = r'%Y-%m-%dT%H:%M:%SZ'
ROLLING_STATISTICS = [
        'total_points', 'ict_index', 'influence', 'creativity', 'threat', 'bonus', 'bps',
        'assists', 'clean_sheets', 'goals_conceded', 'goals_scored', 'minutes', 'saves']


def update_local_data():
    """Updates my local clone of the github.com/vastaav/Fantasy-Premier-League repository."""
    # Get the full repo path.
    path = os.path.abspath('data/Fantasy-Premier-League')
    # This is the command we will run.
    command = f'cd {path} && git pull origin master'
    # Capture output.
    output = subprocess.run(command, capture_output=True, shell=True)
    # Print output.
    print(output.stdout.decode())


def rolling_average_over_games(column, n):
    """Calculates the average value in a column
    over the last n games."""

    # Calculate the rolling average.
    rolling_average = column.rolling(window=n, min_periods=1, closed='left').mean()
    # Fill blanks.
    rolling_average = rolling_average.fillna(0)

    return rolling_average

def rolling_average_over_days(column, n):
    """Calculates the average value in a column
    over the last n days."""

    # For this to work, the index of the column must contain timestamps.

    # Calculate the rolling average.
    rolling_average = column.rolling(window=f'{n}D', min_periods=1, closed='left').mean()
    # Fill blanks.
    rolling_average = rolling_average.fillna(0)
    
    return rolling_average


def get_player_training_data(df):
    
    # Convert the kickoff times to date objects.
    df['kickoff_time'] = pd.to_datetime(df['kickoff_time'], format=DATE_FORMAT)
    # Sort the data by dates.
    df.sort_values('kickoff_time', ignore_index=True, inplace=True)

    # Compute rolling statistics.
    for column in ROLLING_STATISTICS:
        df['avg_' + column + '_5_games'] = rolling_average_over_games(df[column], 5).values
        df['avg_' + column] = rolling_average_over_games(df[column], len(df)).values

    # Check if the player got a red card in their last match.
    df['is_suspended'] = df['red_cards'].shift(1).fillna(0).astype('int')

    return df

def get_all_training_data(season, players, teams, fixtures):

    # Create a list to hold all the data.
    data = []

    # Navigate to the data/season/players directory.
    path = f'data/Fantasy-Premier-League/data/{season}/players'

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
    df['position'] = df.element.map(players.set_index('id').element_type)

    # Map the team stats of both the future opponent and the own team.
    stats = [
        'strength'
        ]
    for stat in stats:
        # Set the stat for the own team.
        df['team_' + stat] = df['team'].map(teams.set_index('id')[stat])
        # Set the stat for the future opponent.
        df['opponent_' + stat] = df['opponent_team'].map(teams.set_index('id')[stat])

    # Drop information we don't need.
    columns_to_drop = [
        'assists', 'bonus', 'bps', 'clean_sheets',
        'creativity', 'goals_conceded', 'goals_scored',
        'ict_index', 'influence', 'kickoff_time', 
        'minutes', 'own_goals', 'penalties_missed',
        'penalties_saved', 'red_cards', 'saves',
        'selected', 'team_a_score', 'team_h_score',
        'threat', 'transfers_balance',
        'transfers_in', 'transfers_out', 'yellow_cards', 
        'team_h', 'team_a', 'team_h_difficulty',
        'team_a_difficulty'
    ]
    df.drop(columns=columns_to_drop, inplace=True)

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
    data.loc['element'] = df.loc[0, 'element']

    # Compute rolling statistics.
    for column in ROLLING_STATISTICS:
        data.loc['avg_' + column + '_5_games'] = df[column].iloc[-5:].mean()
        data.loc['avg_' + column] = df[column].mean()

    # Check if the player got a red card in their last match.
    data.loc['is_suspended'] = df.iloc[-1]['red_cards']

    return data

def get_all_current_data(season, elements, teams, next_gameweek):

    # Create a list to hold all the data.
    data = []

    # Navigate to the data/season/players directory.
    path = f'data/Fantasy-Premier-League/data/{season}/players'

    # Scan through every folder in that directory.
    for item in os.scandir(path):
        
        if item.is_dir():

            gw_path = os.path.join(item.path, 'gw.csv')

            # If the player has no data, skip.
            if not os.path.exists(gw_path):
                continue

            # Read the gw.csv file into a dataframe.
            df = pd.read_csv(gw_path)

            # Get each player's data.
            player_data = get_player_current_data(df, next_gameweek)

            if player_data is None:
                continue

            # Add to the list.
            data.append(player_data)

    # Merge all the data into a single dataframe.
    df = pd.DataFrame(data)

    # Add position information.
    df['position'] = df['element'].map(elements.set_index('id')['element_type'])

    # Add team information.
    df['team'] = df['element'].map(elements.set_index('id')['team'])
    # Map team stats.
    stats = [
        'strength'
        ]
    for stat in stats:
        df['team_' + stat] = df['team'].map(teams.set_index('id')[stat])

    # Rename columns.
    df.rename(columns={'element': 'id'}, inplace=True)

    return df


def main_training_routine():

    print('Starting...')

    seasons = ['2019-20', '2020-21']
    
    for season in seasons:

        # Load accessory files.
        players = pd.read_csv(f'data/Fantasy-Premier-League/data/{season}/players_raw.csv')
        fixtures = pd.read_csv(f'data/Fantasy-Premier-League/data/{season}/fixtures.csv')
        teams = pd.read_csv(f'data/Fantasy-Premier-League/data/{season}/teams.csv')

        # Get training data.
        data = get_all_training_data(season, players, teams, fixtures)

        # Store the season.
        data['season'] = season

        # Save as a CSV file.
        data.to_csv(f'data/{season}.csv', index=False)

        print(f'Done with {season}!')

    input('Done! Press Enter to close.')


def main():
    main_training_routine()
    pass


if __name__ == '__main__':
    main()