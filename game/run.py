import os

import polars as pl
import requests

from features.engineer_features import engineer_match_features, engineer_player_features
from loaders.fpl import load_static_elements, load_static_teams
from loaders.merged import load_merged
from loaders.upcoming import get_upcoming_gameweeks
from loaders.utils import get_mapper, get_seasons
from optimization.optimize import optimize_squad
from optimization.parameters import get_parameters
from prediction.predict import aggregate_predictions, make_predictions, save_predictions
from prediction.utils import load_model


def run(current_season: int, next_gameweek: int, wildcard_gameweeks: list[int]):
    """Run optimization on a live Fantasy Premier League team."""

    parameters = get_parameters()
    fpl_id = int(os.getenv("FPL_ID"))
    fpl_api_authorization = os.getenv("FPL_API_AUTHORIZATION")

    # Load data from disk
    static_elements = load_static_elements(current_season, next_gameweek)
    static_elements = static_elements.collect()
    static_teams = load_static_teams(current_season, next_gameweek)
    static_teams = static_teams.collect()
    seasons = get_seasons(current_season, 2)
    upcoming_gameweeks = get_upcoming_gameweeks(
        next_gameweek, parameters["optimization_window_size"], 38
    )
    players, matches, _ = load_merged(seasons, current_season, upcoming_gameweeks)
    players = players.collect()
    matches = matches.collect()

    # Load team data from the API
    my_team = get_my_team(fpl_id, fpl_api_authorization)
    squad = {pick["element"] for pick in my_team["picks"]}
    budget = my_team["transfers"]["bank"]
    free_transfers = my_team["transfers"]["limit"] or 0
    selling_prices = {
        pick["element"]: pick["selling_price"] for pick in my_team["picks"]
    }

    # Engineer features
    players = engineer_player_features(players)
    matches = engineer_match_features(matches)

    # Keep only the features for upcoming gameweeks
    players = players.filter(
        (pl.col("season") == current_season)
        & (pl.col("gameweek").is_in(upcoming_gameweeks))
    )
    matches = matches.filter(
        (pl.col("season") == current_season)
        & (pl.col("gameweek").is_in(upcoming_gameweeks))
    )

    # Predict total points
    model = load_model("live")
    predictions = make_predictions(model, players, matches)
    save_predictions(predictions, static_elements, static_teams)
    predictions = aggregate_predictions(predictions)

    # Map each prediction to an ID and gameweek
    predictions = get_mapper(predictions, ["element", "gameweek"], "total_points")
    for player in static_elements["id"]:
        for gameweek in upcoming_gameweeks:
            if (player, gameweek) not in predictions:
                predictions[(player, gameweek)] = 0

    # Optimize the squad
    now_costs = get_mapper(static_elements, "id", "now_cost")
    element_types = get_mapper(static_elements, "id", "element_type")
    teams = get_mapper(static_elements, "id", "team")
    web_names = get_mapper(static_elements, "id", "web_name")
    roles = optimize_squad(
        squad,
        budget,
        free_transfers,
        now_costs,
        selling_prices,
        upcoming_gameweeks,
        wildcard_gameweeks,
        predictions,
        element_types,
        teams,
        web_names,
        parameters,
        log=True,
    )

    # Display the optimized squad
    print(roles)


def get_my_team(id: int, api_authorization: str) -> dict:
    """Fetches squad information from FPL."""
    headers = {
        "X-Api-Authorization": api_authorization,
    }
    url = f"https://fantasy.premierleague.com/api/my-team/{id}/"
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()
