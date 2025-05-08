import requests

BASE_URL = "https://fantasy.premierleague.com/api"


def fetch_element_summary(element_id: int) -> dict:
    """Fetches data for a specific FPL player/manager."""
    response = requests.get(f"{BASE_URL}/element-summary/{element_id}/")
    response.raise_for_status()
    return response.json()


def fetch_fixtures() -> dict:
    """Fetches FPL fixture information."""
    response = requests.get(f"{BASE_URL}/fixtures/")
    response.raise_for_status()
    return response.json()


def fetch_bootstrap_static() -> dict:
    """Fetches the current FPL game state."""
    response = requests.get(f"{BASE_URL}/bootstrap-static/")
    response.raise_for_status()
    return response.json()
