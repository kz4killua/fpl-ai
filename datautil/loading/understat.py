"""Functions for loading local data gotten from understat.com"""

import os

import pandas as pd
from pathlib import Path
from datautil.utilities import convert_season_to_year, convert_year_to_season


def load_understat_players():
    """Load local understat.com player data."""

    players = []

    for item in os.scandir('data/understat/player/matches'):
        if item.name.endswith('.csv'):
            
            player = pd.read_csv(item.path)
            player['player_id'] = int(item.name[:-4])

            players.append(player)

    players = pd.concat(players)

    # Add a column for the FPL season format
    players['fpl_season'] = players['season'].apply(convert_year_to_season)
    
    return players


def load_understat_teams(seasons):
    """Load local understat.com team data."""

    teams = []

    for season in seasons:
        understat_season = convert_season_to_year(season)

        path = Path(f'data/understat/season/{understat_season}/teams')
        if not path.exists():
            continue

        for item in os.scandir(path):
            if item.name.endswith('.csv'):

                team = pd.read_csv(item.path)
                team['fpl_season'] = season
                teams.append(team)

    teams = pd.concat(teams, ignore_index=True)

    return teams


def load_understat_fixtures(seasons):
    """Load local fixture data from understat.com"""
    
    fixtures = []

    for season in seasons:
        understat_season = convert_season_to_year(season)

        path = Path(f'data/understat/season/{understat_season}/dates.csv')
        if not path.exists():
            continue

        dates = pd.read_csv(path)
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
        understat_season = convert_season_to_year(season)

        path = Path(f'data/understat/season/{understat_season}/fixture_ids.csv')
        if not path.exists():
            continue

        df = pd.read_csv(path)
        df['fpl_season'] = season
        df['understat_season'] = understat_season

        fixture_ids.append(df)

    fixture_ids = pd.concat(fixture_ids, ignore_index=True)

    return fixture_ids