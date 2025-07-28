import polars as pl

from datautil.upcoming import (
    get_upcoming_fixtures,
    get_upcoming_gameweeks,
    get_upcoming_managers,
    get_upcoming_players,
    get_upcoming_teams,
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
    while simulator.next_gameweek <= simulator.last_gameweek:
        roles = get_gameweek_roles(simulator, model, wildcard_gameweeks, parameters)
        simulator.update(roles, wildcard_gameweeks, log=log)

    return simulator.season_points


def get_gameweek_roles(
    simulator: Simulator,
    model: PredictionModel,
    wildcard_gameweeks: list[int],
    parameters: dict[str, float] | None = None,
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
    upcoming_players = get_upcoming_players(
        upcoming_fixtures, static_players, static_teams
    )
    upcoming_managers = get_upcoming_managers(
        upcoming_fixtures, static_managers, static_teams
    )
    upcoming_teams = get_upcoming_teams(upcoming_fixtures, static_teams)

    # Combine the data for feature engineering
    combined_players = pl.concat(
        [historical_players, upcoming_players], how="diagonal_relaxed"
    )
    combined_teams = pl.concat(
        [historical_teams, upcoming_teams], how="diagonal_relaxed"
    )
    _ = pl.concat([historical_managers, upcoming_managers], how="diagonal_relaxed")

    # Engineer features
    player_features = engineer_player_features(combined_players)
    team_features = engineer_team_features(combined_teams)
    match_features = engineer_match_features(team_features)

    # Keep only the features for upcoming gameweeks
    player_features = player_features.filter(
        (pl.col("season") == season) & (pl.col("round").is_in(upcoming_gameweeks))
    )
    team_features = team_features.filter(
        (pl.col("season") == season) & (pl.col("round").is_in(upcoming_gameweeks))
    )
    match_features = match_features.filter(
        (pl.col("season") == season) & (pl.col("round").is_in(upcoming_gameweeks))
    )

    # Predict total points
    predictions = make_predictions(model, player_features, match_features)
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
        parameters,
    )

    return roles
