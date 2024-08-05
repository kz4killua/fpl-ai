import json
import numpy as np

from .understat.localdata import save_understat_league_dates_data
from .understat.localdata import save_understat_league_teams_data
from .understat.localdata import save_understat_player_matches_data
from .understat.mappings import update_fixture_ids, update_player_ids, update_team_ids
from .fpl import update_local_players, update_local_elements, update_local_teams, update_local_fixtures, update_bootstrap_data

from datautil.constants import LOCAL_DATA_PATH


def update_local_data(season, elements, events, teams):
    """Update local data if it is out of date."""

    checked_gameweeks = events[events['data_checked'] == True]['id']
    if checked_gameweeks.empty:
        last_updated_gameweek = 0
    else:
        last_updated_gameweek = int(checked_gameweeks.max())

    # Check if the local data is already up to date
    checkpoint = LOCAL_DATA_PATH / f"api/{season}/local_players_last_update.json"
    if checkpoint.exists():
        with open(checkpoint, 'r') as f:
            if last_updated_gameweek == json.load(f):
                return

    update_local_players(season, elements)
    update_local_elements(season, elements)
    update_local_teams(season, teams)
    update_local_fixtures(season)
    update_bootstrap_data(season)

    if last_updated_gameweek != 0:
        save_understat_league_dates_data(season)
        save_understat_league_teams_data(season)
        save_understat_player_matches_data(season)
        update_player_ids(season)
        update_team_ids(season)
        update_fixture_ids(season)

    # Keep track of the last updated gameweek
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint, 'w') as f:
        json.dump(last_updated_gameweek, f)