import pandas as pd

from optimize.utilities import get_valid_transfers, make_best_transfer, calculate_budget, update_selling_prices


def run_greedy_optimization(
    squad: set, budget: int, gameweeks: list, now_costs: pd.Series,
    elements: pd.DataFrame, selling_prices: pd.Series,
    gameweek_predictions: pd.Series
):
    """
    Optimizes a squad by replacing each player with their cheapest alternative, 
    then iteratively making the best transfers till convergence. 
    """

    current_squad = squad.copy()
    current_budget = budget
    
    # Replace each player with their cheapest alternative.
    for player in list(current_squad):

        valid_replacements = get_valid_transfers(
            current_squad, player, elements, selling_prices, current_budget
        )
        valid_replacements_costs = now_costs.loc[list(valid_replacements)]
        cheapest_replacement = valid_replacements_costs.idxmin()

        # Update the squad (and budget) accordingly.
        new_squad = current_squad - {player} | {cheapest_replacement}
        new_budget = calculate_budget(
            current_squad, new_squad, current_budget, selling_prices, now_costs
        )
        selling_prices = update_selling_prices(
            selling_prices, now_costs, current_squad, new_squad
        )

        current_squad, current_budget = new_squad, new_budget


    # Iteratively make the single best transfer, until the squad converges.
    while True:
        new_squad = make_best_transfer(
            current_squad, gameweeks, current_budget,
            elements, selling_prices, gameweek_predictions
        )
        new_budget = calculate_budget(
            current_squad, new_squad, 
            current_budget, 
            selling_prices, now_costs
        )
        selling_prices = update_selling_prices(
            selling_prices, now_costs, current_squad, new_squad
        )
        
        if new_squad == current_squad:
            break
        else:
            current_squad = new_squad
            current_budget = new_budget

    return current_squad