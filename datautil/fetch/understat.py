import json
import re

import requests

BASE_URL = "https://understat.com"


def fetch_player_matches(player_id: int) -> dict:
    """Fetches data for a specific player."""
    response = requests.get(f"{BASE_URL}/player/{player_id}/")
    response.raise_for_status()
    return extract_data(response.text, "matchesData")


def fetch_league_dates(league: str, year: int) -> dict:
    """Fetches fixtures for a specific league."""
    response = requests.get(f"{BASE_URL}/league/{league}/{year}/")
    response.raise_for_status()
    return extract_data(response.text, "datesData")


def fetch_league_teams(league: str, year: int) -> dict:
    """Fetches team information for a specific league."""
    response = requests.get(f"{BASE_URL}/league/{league}/{year}/")
    response.raise_for_status()
    return extract_data(response.text, "teamsData")


def fetch_league_players(league: str, year: int) -> dict:
    """Fetches player information for a specific league."""
    response = requests.get(f"{BASE_URL}/league/{league}/{year}/")
    response.raise_for_status()
    return extract_data(response.text, "playersData")


def fetch_match_shots(match_id: int) -> dict:
    """Fetches shot data for a specific match."""
    response = requests.get(f"{BASE_URL}/match/{match_id}/")
    response.raise_for_status()
    return extract_data(response.text, "shotsData")


def extract_data(text: str, name: str) -> dict:
    """Extracts JSON data from a page."""

    # Find matching JSON strings
    pattern = f"{name}.*JSON.parse\('(.*)'\)"
    pattern = re.compile(pattern)
    matches = pattern.findall(text)
    if not matches:
        raise ValueError(f"Could not find JSON data for {name} in the page.")

    # Load the first match as JSON
    match = matches[0]
    match = match.encode("utf-8").decode("unicode_escape")
    return json.loads(match)
