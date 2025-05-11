import base64
import json
import lzma
import re
from datetime import datetime

import requests


def fetch_bootstrap_static(deadline_time: datetime) -> dict:
    """Fetch bootstrap static data from https://github.com/Randdalf/fplcache."""
    tree = get_repository_tree("Randdalf", "fplcache", "main")
    entry = find_cache_entry(tree, deadline_time)
    content = download_blob(entry["url"])
    bootstrap_static = decompress_cache_file(content)
    return bootstrap_static


def find_cache_entry(tree: list, deadline_time: datetime) -> dict:
    """Find the cache entry closest to, but before the deadline time."""

    candidates = []

    # Search the tree for cache files
    pattern = re.compile(r"cache/(\d{4})/(\d{1,2})/(\d{1,2})/(\d{2})(\d{2})\.json\.xz$")
    for entry in tree:
        # Filter out trees and non-cache files
        if entry["type"] != "blob":
            continue
        matches = pattern.match(entry["path"])
        if not matches:
            continue

        # Filter out entries on or after the deadline time
        year, month, day, hour, minute = map(int, matches.groups())
        timestamp = datetime(year, month, day, hour, minute)
        if timestamp >= deadline_time:
            continue

        # Store all valid cache entries
        candidates.append((timestamp, entry))

    # Ensure we have at least one candidate
    if not candidates:
        raise ValueError("No valid cache files found before the deadline time.")

    # Return the entry with the latest timestamp
    timestamp, entry = max(candidates, key=lambda x: x[0])
    return entry


def decompress_cache_file(content: bytes) -> dict:
    """Decompress the cache file content."""
    decompressed = lzma.decompress(content)
    data = json.loads(decompressed.decode("utf-8"))
    return data


def get_repository_tree(owner: str, repo: str, branch: str) -> list:
    """Fetch the file tree for a GitHub repository."""
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()["tree"]


def download_blob(blob_url: str) -> bytes:
    """Download the content of a blob from GitHub."""
    response = requests.get(blob_url)
    response.raise_for_status()
    content = response.json()["content"]
    content = base64.b64decode(content)
    return content
