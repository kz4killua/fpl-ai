"""Functions to add records for predicting future fixtures."""

import numpy as np
import pandas as pd


def insert_fixture_records(season, next_gameweek, fixtures, local_players, local_teams, bootstrap_elements, bootstrap_teams):
    """Add records for predicting future fixtures."""

    for _, fixture in fixtures[fixtures['event'] >= next_gameweek].iterrows():

        # Add player records
        player_matchups = get_player_matchups(season, bootstrap_elements, bootstrap_teams, fixture)
        local_players = pd.concat([local_players, player_matchups])

        # Add team records
        team_matchups = get_team_matchups(season, bootstrap_teams, fixture)
        local_teams = pd.concat([local_teams, team_matchups])

    local_players.reset_index(inplace=True, drop=True)
    local_teams.reset_index(inplace=True, drop=True)

    return local_players, local_teams


def get_player_matchups(season, elements, teams, fixture):
    """Get player data for future fixtures."""

    # Re-index the dataframes to speed up lookups
    elements = elements.set_index('code', drop=False)
    teams = teams.set_index('id', drop=False)

    # Get the codes of all players involved in the fixture
    codes = elements[
        elements['team'].isin([fixture['team_h'], fixture['team_a']])
    ]['code'].values

    # Create a new record for each player involved in the fixture
    records = pd.DataFrame({'code': codes})

    # Fill in feature values
    records['element'] = records['code'].map(elements['id'])
    records['element_type'] = records['code'].map(elements['element_type'])
    records['team'] = records['code'].map(elements['team'])
    records['team_code'] = records['code'].map(elements['team_code'])
    records['fixture'] = fixture['id']
    records['kickoff_time'] = fixture['kickoff_time']
    records['round'] = fixture['event']
    records['was_home'] = fixture['team_h'] == records['team']
    records['season'] = season
    records.loc[records['was_home'] == True, 'opponent_team'] = fixture['team_a']
    records.loc[records['was_home'] == False, 'opponent_team'] = fixture['team_h']
    records['opponent_team_code'] = records['opponent_team'].map(teams['code'])

    return records


def get_team_matchups(season, teams, fixture):
    """Get team data for future fixtures."""

    # Re-index the dataframe to make lookups faster
    teams = teams.set_index('id', drop=False)

    # Add records for both teams involved
    records = [
        pd.Series({
            'h_a': 'h',
            'date': np.datetime64(fixture['kickoff_time']),
            'fpl_season': season,
            'fpl_fixture_id': fixture['id'],
            'fpl_code': teams.loc[fixture['team_h']]['code'],
            'was_home': 1,
        }),
        pd.Series({
            'h_a': 'a',
            'date': np.datetime64(fixture['kickoff_time']),
            'fpl_season': season,
            'fpl_fixture_id': fixture['id'],
            'fpl_code': teams.loc[fixture['team_a']]['code'],
            'was_home': 0,
        })
    ]

    records = pd.concat([
        series.to_frame().T for series in records], ignore_index=True
    )

    return records