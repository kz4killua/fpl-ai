import os
import subprocess
import json
import csv
import difflib

import pandas as pd
import numpy as np

import api
import understat


DATE_FORMAT = r'%Y-%m-%dT%H:%M:%SZ'
ROLLING_STATISTICS = ['total_points']


def year_to_season(year):
    """Returns a string representing a season of format yyyy-yy."""
    return  str(year) + '-' + str(year + 1)[-2:]


def season_to_year(season):
    """Returns the year season 'ssss-ss' started."""
    return int(season[:4])


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


def calculate_similarity(s1: str, s2: str):
    """Returns a score indicating how similar two strings are."""
    # Convert both strings to lowercase
    s1 = s1.lower().strip()
    s2 = s2.lower().strip()
    # Match with difflib
    matcher = difflib.SequenceMatcher(None, s1, s2)
    return matcher.ratio()


def map_closest_names(a: dict, b: dict):
    """
    Maps each name in a.values() with the most similar name in b.values()
    """
    output = dict()
    # Get all possible pairs
    pairs = [(k1, k2) for k1 in a.keys() for k2 in b.keys()]
    # Rank the pairs by similarity
    pairs.sort(key = lambda pair: calculate_similarity(a[pair[0]], b[pair[1]]), reverse=True)
    # Map each key in a to the closest key in b
    mapped_a = set()
    mapped_b = set()

    for pair in pairs:
        if not(pair[0] in mapped_a or pair[1] in mapped_b):
            # Add the mapping
            output[pair[0]] = pair[1]
            # Keep track of what we have already mapped
            mapped_a.add(pair[0])
            mapped_b.add(pair[1])

    return output


def update_player_ids(season):
    """Update FPL to understat mappings for player ids."""

    # Gather players data from the FPL API
    fpl_players = pd.read_csv(f'data/vaastav/data/{season}/players_raw.csv')
    # Gather players data from the understat API
    understat_players = understat.get_league_players_data('EPL', season_to_year(season))
    understat_players = pd.DataFrame(understat_players)
    understat_players['id'] = understat_players['id'].astype('int')
    # Load IDs
    player_ids = pd.read_csv('data/understat/player_ids.csv')

    # Get all FPL players who have not been assigned understat ids
    unmatched_fpl = set(fpl_players['code'].values) - set(player_ids['fpl_code'].values)
    # Filter out all those that have not played a match (they may not have understat data)
    unmatched_fpl &= set(fpl_players[fpl_players['minutes'] != 0]['code'].values)
    # Match their codes to their full names
    unmatched_fpl = dict(
        fpl_players.set_index('code').loc[list(unmatched_fpl)]['first_name'] +
        ' ' +
        fpl_players.set_index('code').loc[list(unmatched_fpl)]['second_name'])
    # Add web names
    for code, full_name in unmatched_fpl.items():
        web_name = fpl_players.set_index('code').loc[code]['web_name']
        if web_name not in full_name:
            unmatched_fpl[code] += ' ' + web_name

    # Get all understat players who have not been assigned FPL codes
    unmatched_understat = understat_players[
        ~understat_players['id'].isin(player_ids['understat_id'].values)
        ]
    # Map their ids to their names
    unmatched_understat = dict(unmatched_understat.set_index('id')['player_name'])

    # Map names
    closest_names = map_closest_names(unmatched_fpl, unmatched_understat)

    with open("data/understat/player_ids.csv", 'a', encoding='utf-8') as f:
        writer = csv.writer(f)
        for pair in closest_names.items():
            fpl_code, understat_id = pair
            fpl_name = unmatched_fpl[fpl_code]
            understat_name = unmatched_understat[understat_id]
            writer.writerow([fpl_code, understat_id, fpl_name, understat_name])


def update_team_ids(season):
    """Update FPL to understat mappings for team ids."""

    # Get FPL team data
    fpl_teams = pd.read_csv(f'data/vaastav/data/{season}/teams.csv')
    # Get understat team data
    understat_teams = understat.get_league_teams_data('EPL', season_to_year(season))
    understat_teams = pd.DataFrame(understat_teams.values())
    understat_teams.drop('history', axis=1, inplace=True)
    understat_teams['id'] = understat_teams['id'].astype('int')
    # Load team IDs
    team_ids = pd.read_csv('data/understat/team_ids.csv')

    # Get all FPL teams who have not been mapped
    unmatched_fpl = set(fpl_teams['code'].values) - set(team_ids['fpl_code'].values)
    # Match their codes to their full names
    unmatched_fpl = dict(
        fpl_teams.set_index('code').loc[list(unmatched_fpl)]['name']
        )

    # Get all understat teams that have not been assigned FPL codes
    unmatched_understat = understat_teams[
        ~understat_teams['id'].isin(team_ids['understat_id'].values)
        ]
    # Map their ids to their names
    unmatched_understat = dict(unmatched_understat.set_index('id')['title'])

    # Map names
    closest_names = map_closest_names(unmatched_fpl, unmatched_understat)

    with open("data/understat/team_ids.csv", 'a', encoding='utf-8') as f:
        writer = csv.writer(f)
        for pair in closest_names.items():
            fpl_code, understat_id = pair
            fpl_name = unmatched_fpl[fpl_code]
            understat_name = unmatched_understat[understat_id]
            writer.writerow([fpl_code, understat_id, fpl_name, understat_name])
    

def update_fixture_ids(season):
    """Update FPL to understat mappings for fixture ids."""

    # Load FPL fixtures for that season
    fpl_fixtures = pd.read_csv(f'data/vaastav/data/{season}/fixtures.csv')

    # Get all understat fixtures for that season
    understat_fixtures = understat.get_league_dates_data("EPL", season_to_year(season))

    # Get home and away ids
    for fixture in understat_fixtures:
        fixture['h'] = fixture['h']['id']
        fixture['a'] = fixture['a']['id']

    # Make a data frame for understat fixtures
    understat_fixtures = pd.DataFrame(understat_fixtures)

    # Convert data types
    understat_fixtures['id'] = understat_fixtures['id'].astype('int')
    understat_fixtures['h'] = understat_fixtures['h'].astype('int')
    understat_fixtures['a'] = understat_fixtures['a'].astype('int')

    # Load team ids
    team_ids = pd.read_csv('data/understat/team_ids.csv')
    # Load team data for that season
    teams = pd.read_csv(f'data/vaastav/data/{season}/teams.csv')

    # Map fpl codes for home and away teams
    fpl_fixtures['team_h_fpl_code'] = fpl_fixtures['team_h'].map(teams.set_index('id')['code'])
    fpl_fixtures['team_a_fpl_code'] = fpl_fixtures['team_a'].map(teams.set_index('id')['code'])
    # Do the same for understat
    understat_fixtures['team_h_fpl_code'] = understat_fixtures['h'].map(team_ids.set_index('understat_id')['fpl_code'])
    understat_fixtures['team_a_fpl_code'] = understat_fixtures['a'].map(team_ids.set_index('understat_id')['fpl_code'])

    understat_fixtures.set_index(['team_h_fpl_code', 'team_a_fpl_code'], inplace=True)
    fpl_fixtures.set_index(['team_h_fpl_code', 'team_a_fpl_code'], inplace=True)

    # Now, match each understat fixture to its FPL code
    understat_fixtures['fpl_id'] = understat_fixtures.index.map(fpl_fixtures['id'])

    # Cleanup
    understat_fixtures.reset_index(inplace=True)
    understat_fixtures.rename({'id': 'understat_id'}, axis=1, inplace=True)

    # Get the columns we need
    fixture_ids = understat_fixtures

    fixture_ids['h_fpl'] = fixture_ids['team_h_fpl_code'].map(
        team_ids.set_index('fpl_code')['fpl_name'])
    fixture_ids['a_fpl'] = fixture_ids['team_a_fpl_code'].map(
        team_ids.set_index('fpl_code')['fpl_name'])

    fixture_ids['h_understat'] = fixture_ids['h'].map(
        team_ids.set_index('understat_id')['understat_name'])
    fixture_ids['a_understat'] = fixture_ids['a'].map(
        team_ids.set_index('understat_id')['understat_name'])

    fixture_ids = fixture_ids[[
        'fpl_id', 'understat_id', 'h_fpl', 'a_fpl', 
        'h_understat', 'a_understat']]

    # Save
    fixture_ids.to_csv(f'data/understat/season/{season_to_year(season)}/fixture_ids.csv', index=False)


def save_understat_player_matches_data(season):
    """Saves understat data for all players who played in a season"""

    # Get ids of all understat players in that league
    players_data = understat.get_league_players_data('EPL', season_to_year(season))
    understat_ids = pd.DataFrame(players_data)['id'].astype('int').values

    for id in understat_ids:
        player_matches_data = understat.get_player_matches_data(id)
        # Save to a CSV
        player_matches_data = pd.DataFrame(player_matches_data)
        player_matches_data.to_csv(f'data/understat/player/matches/{id}.csv', index=False)


def save_understat_league_teams_data(season):
    # Get all the data
    teams_data = understat.get_league_teams_data('EPL', season_to_year(season))

    for team in teams_data.values():
        # Create dataframe
        history = pd.DataFrame(team['history'])
        history['id'] = team['id']
        history['title'] = team['title']
        # Save
        history.to_csv(f"data/understat/season/{season_to_year(season)}/teams/{team['id']}.csv", index=False)




def main():
    pass

if __name__ == '__main__':
    main()