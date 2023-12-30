import requests


def get_player_data(element):
    """Returns a player's detailed information in the current season."""
    url = f"https://fantasy.premierleague.com/api/element-summary/{element}/"
    return requests.get(url).json()


def get_fixture_data():
    """Returns fixture information for the current season."""
    url = "https://fantasy.premierleague.com/api/fixtures/"
    return requests.get(url).json()


def get_bootstrap_data():
    """Returns bootstrap data for the current season."""
    url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
    return requests.get(url).json()