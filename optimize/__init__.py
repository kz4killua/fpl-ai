import pandas as pd

from optimize.utilities import get_future_gameweeks, make_best_transfer
from optimize.greedy import run_greedy_optimization


def optimize_squad(
    current_season: str, current_squad: set, current_budget: int, 
    next_gameweek: int, wildcard_gameweeks: list,
    now_costs: pd.Series, selling_prices: pd.Series, 
    gameweek_elements: pd.DataFrame, gameweek_predictions: pd.Series
):
    """Make transfers to optimize the squad for the next gameweek."""

    future_gameweeks = get_future_gameweeks(next_gameweek, wildcard_gameweeks=wildcard_gameweeks)

    if (current_season == '2022-23') and (7 in future_gameweeks):
        future_gameweeks.remove(7)

    # Make multiple transfers on GW 1 and wildcard gameweeks
    if (next_gameweek == 1) or (next_gameweek in wildcard_gameweeks):
        best_squad = run_greedy_optimization(
            current_squad, current_budget, future_gameweeks, 
            now_costs, gameweek_elements, selling_prices, gameweek_predictions,
        )
    # On other gameweeks, make the single best transfer
    else:
        best_squad = make_best_transfer(
            current_squad, future_gameweeks, current_budget, 
            gameweek_elements, selling_prices, now_costs, gameweek_predictions,
        )

    return best_squad