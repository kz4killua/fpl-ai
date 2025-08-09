import polars as pl

from datautil.upcoming import (
    get_upcoming_gameweeks,
)
from datautil.utils import get_mapper
from features.engineer_features import (
    engineer_match_features,
    engineer_player_features,
    engineer_team_features,
)
from optimization.optimize import optimize_squad
from optimization.parameters import OPTIMIZATION_WINDOW_SIZE
from prediction.model import PredictionModel
from prediction.predict import make_predictions
from prediction.utils import load_model

from .simulator import Simulator
from .utils import get_selling_prices


def simulate(
    season: str,
    wildcard_gameweeks: list[int],
    parameters: dict[str, float] | None = None,
    log: bool = False,
) -> int:
    model = load_model(f"simulation_{season}")
    simulator = Simulator(season)

    # Simulate each gameweek
    while simulator.next_gameweek is not None:
        roles = get_best_roles(simulator, model, wildcard_gameweeks, parameters, log)
        simulator.update(roles, wildcard_gameweeks, log=log)

    return simulator.season_points


def get_best_roles(
    simulator: Simulator,
    model: PredictionModel,
    wildcard_gameweeks: list[int],
    parameters: dict[str, float] | None = None,
    log: bool = False,
) -> dict:
    """Get the optimal squad roles for the next gameweek."""
    # Unpack data from the simulator
    season = simulator.season
    squad = simulator.squad
    purchase_prices = simulator.purchase_prices
    budget = simulator.budget
    free_transfers = simulator.free_transfers
    next_gameweek = simulator.next_gameweek
    last_gameweek = simulator.last_gameweek
    static_elements = simulator.static_elements
    players = simulator.players
    teams = simulator.teams
    matches = simulator.managers

    # Engineer features
    players = engineer_player_features(players)
    teams = engineer_team_features(teams)
    matches = engineer_match_features(teams)

    # Keep only the features for upcoming gameweeks
    upcoming_gameweeks = get_upcoming_gameweeks(
        next_gameweek, OPTIMIZATION_WINDOW_SIZE, last_gameweek
    )
    players = players.filter(
        (pl.col("season") == season) & (pl.col("round").is_in(upcoming_gameweeks))
    )
    teams = teams.filter(
        (pl.col("season") == season) & (pl.col("round").is_in(upcoming_gameweeks))
    )
    matches = matches.filter(
        (pl.col("season") == season) & (pl.col("round").is_in(upcoming_gameweeks))
    )

    # Predict total points
    predictions = make_predictions(model, players, matches)

    # Map each prediction to an ID and gameweek
    predictions = get_mapper(predictions, ["element", "round"], "total_points")
    for player in static_elements["id"]:
        for gameweek in upcoming_gameweeks:
            if (player, gameweek) not in predictions:
                predictions[(player, gameweek)] = 0

    # Optimize the squad
    now_costs = get_mapper(static_elements, "id", "now_cost")
    element_types = get_mapper(static_elements, "id", "element_type")
    teams = get_mapper(static_elements, "id", "team")
    selling_prices = get_selling_prices(squad, purchase_prices, now_costs)
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
        log,
    )

    return roles
