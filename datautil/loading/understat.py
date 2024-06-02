"""Functions for loading local data gotten from understat.com"""

import os

import pandas as pd


def load_understat_players():
    """Load local understat.com player data."""

    players = []

    for item in os.scandir('data/understat/player/matches'):
        if item.name.endswith('.csv'):
            
            # Read each player's data
            player = pd.read_csv(item.path)

            # Add general player information
            player['player_id'] = int(item.name[:-4])

            # Save to the list
            players.append(player)

    # Concatenate all players
    players = pd.concat(players)

    # Add a column for the FPL season format
    players['fpl_season'] = players['season'].apply(understat_season_to_fpl_season)
    
    return players


def load_understat_teams(seasons):
    """Load local understat.com team data."""

    teams = []

    for season in seasons:
        season = season[:4]

        for item in os.scandir(f'data/understat/season/{season}/teams'):
            if item.name.endswith('.csv'):

                # Read each team's data
                team = pd.read_csv(item.path)

                # Add general information
                team['fpl_season'] = understat_season_to_fpl_season(season)

                # Add to the list
                teams.append(team)

    # Concatenate all teams
    teams = pd.concat(teams, ignore_index=True)

    return teams


def load_understat_fixtures(seasons):
    """Load local fixture data from understat.com"""
    
    fixtures = []

    for season in seasons:
        dates = pd.read_csv(f'data/understat/season/{season[:4]}/dates.csv')

        dates['fpl_season'] = season
        
        fixtures.append(dates)

    fixtures = pd.concat(fixtures, ignore_index=True)

    return fixtures


def load_player_ids():
    """Load FPL-understat player ID mappings."""
    return pd.read_csv('data/understat/player_ids.csv')


def load_fixture_ids(seasons):
    """Load FPL-understat fixture mappings."""

    fixture_ids = []

    for season in seasons:

        understat_season = int(season[:4])
        
        # Load each season's fixture ID mappings
        df = pd.read_csv(f'data/understat/season/{understat_season}/fixture_ids.csv')

        # Add extra helpful information
        df['fpl_season'] = season
        df['understat_season'] = understat_season

        fixture_ids.append(df)

    fixture_ids = pd.concat(fixture_ids, ignore_index=True)

    return fixture_ids


def understat_season_to_fpl_season(season):
    """Format a season as yyyy-yy."""
    return  str(season) + '-' + str(int(season) + 1)[-2:]