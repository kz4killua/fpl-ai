import json
import re

import requests


LEAGUES = ['EPL', 'La_liga', 'Bundesliga', 'Serie_A', 'Ligue_1', 'RFPL']

BASE_URL = 'https://understat.com/'

JSON_TEMPLATE = "{}.*JSON.parse\('(.*)'\)"


def get_match_url(id):
    return f'{BASE_URL}match/{id}/'


def get_league_url(league, season):
    return f'{BASE_URL}league/{league}/{season}/'


def get_player_url(id):
    return f'{BASE_URL}player/{id}/'


def fetch_jsons(text, name):
    """Returns all embedded JSONs with the name 'name' in a page"""
    regex = JSON_TEMPLATE.format(name)
    regex = re.compile(regex)
    return regex.findall(text)


def load_json(data):
    """Loads a JSON string"""
    data = data.encode('utf-8').decode('unicode_escape')
    data = json.loads(data)
    return data


def get_page_data(text, name):
    """Gets data from a page."""
    data = fetch_jsons(text, name)[0]
    data = load_json(data)
    return data


def get_player_matches_data(id):
    """Gets matches data for a player."""
    r = requests.get(get_player_url(id))
    return get_page_data(r.text, 'matchesData')


def get_league_dates_data(league, year):
    """Gets fixture data for a league."""
    r = requests.get(get_league_url(league, year))
    return get_page_data(r.text, 'datesData')


def get_league_teams_data(league, year):
    """Gets teams data for a league."""
    r = requests.get(get_league_url(league, year))
    return get_page_data(r.text, 'teamsData')


def get_league_players_data(league, year):
    """Gets players data for a league."""
    r = requests.get(get_league_url(league, year))
    return get_page_data(r.text, 'playersData')


def get_match_shots_data(id):
    """Gets shots data for a match."""
    r = requests.get(get_match_url(id))
    return get_page_data(r.text, 'shotsData')