"""Functions for loading local data gotten from the Fantasy Premier League API"""

import os

import pandas as pd


def load_fpl_players(seasons):
    """Load local Fantasy Premier League player data."""

    players = []

    for season in seasons:

        # Load general player information
        elements = pd.read_csv(f'data/api/{season}/elements.csv')
        elements = elements.set_index('id')

        # Load fixture information
        fixtures = pd.read_csv(f'data/api/{season}/fixtures.csv')
        fixtures = fixtures.set_index('id')

        # Load team information
        teams = pd.read_csv(f'data/api/{season}/teams.csv')
        teams = teams.set_index('id')

        for item in os.scandir(f'data/api/{season}/players'):
            if item.name.endswith('.csv'):

                # Read each player's data
                player = pd.read_csv(item.path)

                # Add season information
                player['season'] = season

                # Add general player information
                player['code'] = player['element'].map(
                    elements['code']
                )
                player['element_type'] = player['element'].map(
                    elements['element_type']
                )

                # Add team information
                player['team_h'] = player['fixture'].map(fixtures['team_h'])
                player['team_a'] = player['fixture'].map(fixtures['team_a'])

                home = player[player['was_home'] == True]
                away = player[player['was_home'] == False]

                player.loc[home.index, 'team'] = home['team_h']
                player.loc[away.index, 'team'] = away['team_a']

                player['team_code'] = player['team'].map(
                    teams['code']
                )
                player['opponent_team_code'] = player['opponent_team'].map(
                    teams['code']
                )
                
                # Save to the list
                players.append(player)
    
    # Concatenate all players
    players = pd.concat(players, ignore_index=True)
    
    return players


def load_fpl_teams(seasons):
    """Load all local Fantasy Premier League team data."""

    teams = []
    
    for season in seasons:
        team = pd.read_csv(f'data/api/{season}/teams.csv')

        # Add to the list
        teams.append(team)

    # Concatenate
    teams = pd.concat(teams, ignore_index=True)

    return teams