
import polars as pl

from datautil.upcoming import (
    get_upcoming_fixtures,
    get_upcoming_gameweeks,
    get_upcoming_manager_data,
    get_upcoming_player_data,
    get_upcoming_team_data,
)
from datautil.utils import get_mapper
from features.engineer_features import engineer_features
from optimization.optimize import optimize_squad
from optimization.parameters import OPTIMIZATION_WINDOW_SIZE
from prediction.model import load_model
from prediction.predict import make_predictions

from .simulator import Simulator
from .utils import get_selling_prices


def simulate(season: str, wildcard_gameweeks: list[int], log: bool = False) -> int:
    model = load_model("simulation_model")
    simulator = Simulator(season)

    # Simulate each gameweek
    while simulator.next_gameweek <= simulator.last_gameweek:
        # Unpack attributes from the simulator
        squad = simulator.squad
        purchase_prices = simulator.purchase_prices
        budget = simulator.budget
        free_transfers = simulator.free_transfers
        next_gameweek = simulator.next_gameweek
        last_gameweek = simulator.last_gameweek

        # Load data from the simulator
        static_elements = simulator.static_elements
        static_players = simulator.static_players
        static_teams = simulator.static_teams
        static_managers = simulator.static_managers
        historical_players = simulator.historical_players
        historical_teams = simulator.historical_teams
        historical_managers = simulator.historical_managers
        fixtures = simulator.fixtures

        # Get upcoming data
        upcoming_gameweeks = get_upcoming_gameweeks(
            next_gameweek, OPTIMIZATION_WINDOW_SIZE, last_gameweek
        )
        upcoming_fixtures = get_upcoming_fixtures(fixtures, season, upcoming_gameweeks)
        upcoming_players = get_upcoming_player_data(
            upcoming_fixtures, static_players, static_teams
        )
        upcoming_managers = get_upcoming_manager_data(
            upcoming_fixtures, static_managers, static_teams
        )
        upcoming_teams = get_upcoming_team_data(upcoming_fixtures, static_teams)

        # Combine the data
        combined_players = pl.concat(
            [historical_players, upcoming_players], how="diagonal_relaxed"
        )
        _ = pl.concat([historical_teams, upcoming_teams], how="diagonal_relaxed")
        _ = pl.concat([historical_managers, upcoming_managers], how="diagonal_relaxed")

        # Engineer features. Keep only those needed for prediction
        features = engineer_features(combined_players)
        features = features.filter(
            (pl.col("season") == season) & (pl.col("round").is_in(upcoming_gameweeks))
        )

        # Collect all lazy frames
        static_elements = static_elements.collect()
        features = features.collect()

        # Predict total points
        predictions = make_predictions(features, model)
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
        )

        # Update the simulator
        simulator.update(roles, wildcard_gameweeks, log=log)

    return simulator.season_points
