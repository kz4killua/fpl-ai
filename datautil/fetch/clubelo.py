from csv import DictReader
from io import StringIO

import requests

BASE_URL = "http://api.clubelo.com"


def fetch_rating_history(club_name: str) -> list[dict]:
    """Fetches the rating history for a specific club."""
    response = requests.get(f"{BASE_URL}/{club_name.replace(' ', '')}")
    response.raise_for_status()
    reader = DictReader(StringIO(response.text))
    return list(reader)
