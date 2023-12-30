"""Functions for merging local data from Fantasy Premier League and understat.com"""

import json

import pandas as pd


def merge_players(fpl_players, understat_players, player_ids, fixture_ids):
    """Merge player data from Fantasy Premier League and understat.com."""
    
    # Give each understat player their FPL code
    understat_players['fpl_code'] = understat_players['player_id'].map(
        player_ids.set_index('understat_id')['fpl_code']
    )

    # Add FPL fixture IDs
    understat_players['fpl_fixture_id'] = understat_players.set_index(['fpl_season', 'id']).index.map(
        fixture_ids.set_index(['fpl_season', 'understat_id'])['fpl_id']
    )

    # Any unmapped fixtures are for matches outside the Premier League. Remove them.
    understat_players.dropna(subset=['fpl_fixture_id'], axis=0, inplace=True)

    # Reindex (once) to make mapping faster
    fpl_players = fpl_players.set_index(['code', 'season', 'fixture'], drop=False)
    understat_players = understat_players.set_index(['fpl_code', 'fpl_season', 'fpl_fixture_id'], drop=False)

    # Match understats by FPL code, season, and fixture ID
    for column in [
        'shots', 'xG', 'position', 'xA', 'key_passes', 'npg', 'npxG', 'xGChain', 'xGBuildup'
    ]:
        fpl_players[f'understat_{column}'] = fpl_players.index.map(
            understat_players[column]
        )

    # Restore indices
    fpl_players = fpl_players.reset_index(drop=True)
    understat_players = understat_players.reset_index(drop=True)

    return fpl_players


def merge_teams(understat_teams, understat_fixtures, fixture_ids):
    """Merge team data from Fantasy Premier League and understat.com."""

    # Expand PPDA stats for understat teams
    understat_teams[['ppda_att', 'ppda_def']] = understat_teams['ppda'].apply(extract_ppda_items)
    understat_teams[['ppda_allowed_att', 'ppda_allowed_def']] = understat_teams['ppda_allowed'].apply(extract_ppda_items)
    understat_teams = understat_teams.drop(columns=['ppda', 'ppda_allowed'])

    # Add understat fixture ID
    home = understat_teams[understat_teams['h_a'] == 'h']
    away = understat_teams[understat_teams['h_a'] == 'a']
    understat_teams.loc[home.index, 'fixture_id'] = home.set_index(['id', 'date']).index.map(
        understat_fixtures.set_index(['h', 'datetime'])['id'])
    understat_teams.loc[away.index, 'fixture_id'] = away.set_index(['id', 'date']).index.map(
        understat_fixtures.set_index(['a', 'datetime'])['id'])
    
    # Add FPL fixture IDs
    understat_teams['fpl_fixture_id'] = understat_teams.set_index(['fpl_season', 'fixture_id']).index.map(
        fixture_ids.set_index(['fpl_season', 'understat_id'])['fpl_id']
    )

    # Add FPL team codes
    team_ids = pd.read_csv('data/understat/team_ids.csv')
    understat_teams['fpl_code'] = understat_teams['id'].map(
        team_ids.set_index('understat_id')['fpl_code']
    )

    return understat_teams


def extract_ppda_items(string):
    """Turn a JSON string into a series."""
    data = json.loads(string.replace("'", "\""))
    return pd.Series(data)