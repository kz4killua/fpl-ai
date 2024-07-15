import json
from pathlib import Path

import pandas as pd
import numpy as np

from datautil.constants import LOCAL_DATA_PATH
from datautil.pipeline import load_players_and_teams, insert_fixture_records
from datautil.utilities import get_previous_seasons
from features.features import engineer_features
from predictions import make_predictions


def load_simulation_true_results(season: str, use_cache=True) -> pd.DataFrame:
    """Get total minutes and points for each player in each round of the season."""

    cache_path = Path(f'cache/simulation/true-results-{season}.pkl')
    if use_cache and cache_path.exists():
        return pd.read_pickle(cache_path)

    # Sum up total_points and minutes for each player in each round
    local_players, _ = load_players_and_teams([season])
    season_players = local_players[local_players['season'] == season]
    columns = ['element', 'round', 'total_points', 'minutes']
    true_results = season_players[columns].groupby(['element', 'round']).sum()

    if use_cache:
        true_results.to_pickle(cache_path)

    return true_results


def load_simulation_fixtures(season: str) -> pd.DataFrame:
    """
    Load fixture data for a given season.
    
    This does not account for fixture changes that may occur during the season.
    """
    fixtures = pd.read_csv(LOCAL_DATA_PATH / f"api/{season}/fixtures.csv")
    fixtures['kickoff_time'] = pd.to_datetime(fixtures['kickoff_time'])
    return fixtures


def load_simulation_purchase_prices(season: str, squad: set, next_gameweek: int) -> pd.Series:
    """Load purchase prices for the next gameweek."""
    elements = load_simulation_bootstrap_elements(season, next_gameweek)
    purchase_prices = elements['now_cost'].loc[list(squad)]
    return purchase_prices


def load_simulation_bootstrap(season: str, next_gameweek: int) -> dict:
    with open(LOCAL_DATA_PATH / f"api/{season}/bootstrap/after_gameweek_{next_gameweek-1}.json") as f:
        bootstrap = json.load(f)
    return bootstrap


def load_simulation_bootstrap_elements(season: str, next_gameweek: int):
    """Load `elements` data for the next gameweek."""

    bootstrap = load_simulation_bootstrap(season, next_gameweek)
    elements = pd.DataFrame(bootstrap['elements'])
    elements.set_index('id', inplace=True, drop=False)
    elements['chance_of_playing_next_round'].fillna(100, inplace=True)

    return elements


def load_simulation_bootstrap_teams(season: str, next_gameweek: int):
    """Load team data for the next gameweek."""

    bootstrap = load_simulation_bootstrap(season, next_gameweek)
    teams = pd.DataFrame(bootstrap['teams'])
    return teams


def load_simulation_players_and_teams(season: str, next_gameweek: int):
    """Load `local_players` and `local_teams` for the simulation."""

    previous_seasons = get_previous_seasons(season)
    assert season in previous_seasons
    local_players, local_teams = load_players_and_teams(previous_seasons)

    # Filter out records not in previous seasons
    local_players = local_players[
        local_players['season'].isin(previous_seasons)
    ]
    local_teams = local_teams[
        local_teams['fpl_season'].isin(previous_seasons)
    ]

    # Filter out records not before the next gameweek
    local_players = local_players[
        (local_players['season'] != season) |
        (local_players['round'] < next_gameweek)
    ]
    local_teams = local_teams[
        (local_teams['fpl_season'] != season) |
        (
            local_teams['fpl_fixture_id'].isin(
                local_players[local_players['season'] == season]['fixture']
            )
        )
    ]

    return local_players, local_teams


def load_simulation_features(season: str, next_gameweek: int, use_cache=True):
    """Get prediction features for the next gameweek."""

    cache_path = Path(f'cache/simulation/features-{season}-{next_gameweek}.pkl')

    # Load predictions from cache (if available)
    if use_cache and cache_path.exists():
        return pd.read_pickle(cache_path)
    
    # Load local data and engineer features
    fixtures = load_simulation_fixtures(season)
    bootstrap_elements = load_simulation_bootstrap_elements(season, next_gameweek)
    bootstrap_teams = load_simulation_bootstrap_teams(season, next_gameweek)
    local_players, local_teams = load_simulation_players_and_teams(season, next_gameweek)
    local_players, local_teams = insert_fixture_records(
        season, next_gameweek, fixtures, local_players, local_teams, bootstrap_elements, bootstrap_teams
    )
    features, _ = engineer_features(local_players, local_teams)
    features = features[features['season'] == season]

    # Save features to cache (if requested)
    if use_cache:
        features.to_pickle(cache_path)

    return features