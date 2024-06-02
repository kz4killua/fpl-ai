"""Functions to prepare data for feature extraction and modeling"""

import pandas as pd
import numpy as np


def wrangle_players(players):
    """Handle missing values, convert dtypes, etc."""

    # Remove all columns not in the most recent season
    missing = players.columns[
        players[players['season'] == players['season'].max()].count() == 0
    ]
    players = players.drop(columns=missing)

    # Fill in missing values for understats (except understat_position)
    columns = [
        'understat_xGBuildup', 'understat_npxG', 'understat_xG',
        'understat_xG', 'understat_xGChain', 'understat_xA',
        'understat_shots', 'understat_key_passes', 'understat_npg',
    ]
    players = players.fillna(value={c: 0 for c in columns})

    # Fill in missing values in 'understat_position'
    players = players.fillna(value={'understat_position': 'Reserves'})

    # Convert 'was_home' to int objects
    players['was_home'] = players['was_home'].astype(int)

    # Convert dates to datetime objects
    players['kickoff_time'] = pd.to_datetime(players['kickoff_time'])

    # Sort players by kickoff times
    players = players.sort_values('kickoff_time', ignore_index=True)

    return players


def wrangle_teams(teams):
    """Handle missing values, convert dtypes, etc."""

    # Convert dates to datetime objects
    teams['date'] = teams['date'].apply(np.datetime64)

    # Convert `h_a` to a boolean.
    teams['was_home'] = (teams['h_a'] == 'h').astype(int)

    # Sort teams by kickoff times
    teams = teams.sort_values('date', ignore_index=True)
    
    return teams