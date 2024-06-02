import csv
import difflib

import pandas as pd

from datautil.constants import LOCAL_DATA_PATH
from api.understat import get_league_players_data, get_league_teams_data, get_league_dates_data
from .utilities import season_to_year


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
    """Update FPL to understat mappings for player IDs."""

    # Load general FPL player data
    fpl_players = pd.read_csv(LOCAL_DATA_PATH / f"api/{season}/elements.csv")

    # Load understat player data
    understat_players = get_league_players_data('EPL', season_to_year(season))
    understat_players = pd.DataFrame(understat_players)

    # Correct ID data type
    understat_players['id'] = understat_players['id'].astype('int')

    # Load existing player ID mappings
    player_ids = pd.read_csv(LOCAL_DATA_PATH / 'understat/player_ids.csv')

    # Get all FPL players who have not been assigned understat IDs
    unmatched_fpl = set(fpl_players['code'].values) - set(player_ids['fpl_code'].values)

    # Filter out all those that have not played a match (they may not have understat data)
    unmatched_fpl &= set(fpl_players[fpl_players['minutes'] != 0]['code'].values)

    # Match their codes to their full names
    first_name = fpl_players.set_index('code').loc[list(unmatched_fpl)]['first_name']
    second_name = fpl_players.set_index('code').loc[list(unmatched_fpl)]['second_name']
    unmatched_fpl = dict(first_name + ' ' + second_name)

    # Add web names (for similarities)
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

    # Map names together
    closest_names = map_closest_names(unmatched_fpl, unmatched_understat)

    # Save CSV
    with open(LOCAL_DATA_PATH / "understat/player_ids.csv", 'a', encoding='utf-8') as f:
        writer = csv.writer(f)
        for pair in closest_names.items():
            fpl_code, understat_id = pair
            fpl_name = unmatched_fpl[fpl_code]
            understat_name = unmatched_understat[understat_id]
            writer.writerow([fpl_code, understat_id, fpl_name, understat_name])


def update_team_ids(season):
    """Update FPL to understat mappings for team ids."""

    # Load FPL team data
    fpl_teams = pd.read_csv(LOCAL_DATA_PATH / f"api/{season}/teams.csv")

    # Get understat team data
    understat_teams = get_league_teams_data('EPL', season_to_year(season))
    understat_teams = pd.DataFrame(understat_teams.values())

    # Drop unnecessary columns
    understat_teams.drop('history', axis=1, inplace=True)

    # Fix data types
    understat_teams['id'] = understat_teams['id'].astype('int')

    # Load existing team IDs
    team_ids = pd.read_csv(LOCAL_DATA_PATH / 'understat/team_ids.csv')

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

    # Save CSV
    with open(LOCAL_DATA_PATH / "understat/team_ids.csv", 'a', encoding='utf-8') as f:
        writer = csv.writer(f)
        for pair in closest_names.items():
            fpl_code, understat_id = pair
            fpl_name = unmatched_fpl[fpl_code]
            understat_name = unmatched_understat[understat_id]
            writer.writerow([fpl_code, understat_id, fpl_name, understat_name])


def update_fixture_ids(season):
    """Update FPL to understat mappings for fixture ids."""

    # Load FPL fixtures for that season
    fpl_fixtures = pd.read_csv(LOCAL_DATA_PATH / f'api/{season}/fixtures.csv')

    # Get all understat fixtures for that season
    understat_fixtures = get_league_dates_data("EPL", season_to_year(season))

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
    team_ids = pd.read_csv(LOCAL_DATA_PATH / 'understat/team_ids.csv')
    # Load team data for that season
    teams = pd.read_csv(LOCAL_DATA_PATH / f'api/{season}/teams.csv')

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

    # Create the folder to store the data
    path = LOCAL_DATA_PATH / f'understat/season/{season_to_year(season)}/fixture_ids.csv'
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save to a CSV
    fixture_ids.to_csv(path, index=False)