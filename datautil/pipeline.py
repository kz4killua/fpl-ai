"""End-to-end data loading pipelines for players and teams"""

from .loading.fpl import load_fpl_players, load_fpl_teams
from .loading.understat import load_understat_players, load_fixture_ids, load_player_ids, load_understat_fixtures, load_understat_teams
from .merging import merge_players, merge_teams
from .wrangling import wrangle_players, wrangle_teams
from .injecting import insert_fixture_records


def load_players_and_teams(seasons, fixtures, elements, teams):

    local_players = load_players(seasons)
    local_teams = load_teams(seasons)
    local_players, local_teams = insert_fixture_records(fixtures, local_players, local_teams, max(seasons), elements, teams)

    return local_players, local_teams


def load_players(seasons):
    """Load, merge, and clean player data from all sources."""

    # Load Fantasy Premier League (API) player data
    fpl_players = load_fpl_players(seasons)

    # Load understat.com player data
    understat_players = load_understat_players()

    # Load player and fixture ID mappings
    fixture_ids = load_fixture_ids(seasons)
    player_ids = load_player_ids()

    # Merge all player data
    players = merge_players(fpl_players, understat_players, player_ids, fixture_ids)

    # Handle missing values, convert data types, etc.
    players = wrangle_players(players)

    return players


def load_teams(seasons):
    """Load, merge, and clean team data from all sources."""

    # Load Fantasy Premier League (API) team data
    fpl_teams = load_fpl_teams(seasons)

    # Load all understat.com team data
    understat_teams = load_understat_teams(seasons)

    # Load understat.com fixtures and fixture ID mappings
    understat_fixtures = load_understat_fixtures(seasons)
    fixture_ids = load_fixture_ids(seasons)

    # Merge all team data
    teams = merge_teams(understat_teams, understat_fixtures, fixture_ids)

    # Handle missing values, convert data types, etc.
    teams = wrangle_teams(teams)

    return teams