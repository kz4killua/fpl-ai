from .understat.localdata import save_understat_league_dates_data
from .understat.localdata import save_understat_league_teams_data
from .understat.localdata import save_understat_player_matches_data
from .understat.mappings import update_fixture_ids, update_player_ids, update_team_ids
from .fpl import update_local_players, update_local_elements, update_local_teams, update_local_fixtures


def update_local_data(season, elements, events, teams):
    update_local_players(season, elements, events)
    update_local_elements(season, elements)
    update_local_teams(season, teams)
    update_local_fixtures(season)
    save_understat_league_dates_data(season)
    save_understat_league_teams_data(season)
    save_understat_player_matches_data(season)
    update_player_ids(season)
    update_team_ids(season)
    update_fixture_ids(season)