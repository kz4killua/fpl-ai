import pandas as pd
import numpy as np

from optimize.utilities import make_best_transfer, suggest_squad_roles, get_future_gameweeks, calculate_points, calculate_budget, sum_player_points
from predictions import make_predictions, group_predictions_by_gameweek, weight_gameweek_predictions_by_availability
from simulation.utilities import make_automatic_substitutions, get_selling_prices, update_purchase_prices, update_selling_prices
from optimize.greedy import run_greedy_optimization
from simulation.utilities import make_automatic_substitutions, get_selling_prices, update_purchase_prices, update_selling_prices
from simulation.loaders import load_simulation_purchase_prices, load_simulation_bootstrap_elements, load_simulation_features, load_simulation_true_results


def get_full_name(player_id: int, elements: pd.DataFrame):
    """Returns a player's full name, formatted as `first second (web)`."""
    first_name = elements.loc[player_id, 'first_name']
    second_name = elements.loc[player_id, 'second_name']
    web_name = elements.loc[player_id, 'web_name']
    return f"{first_name} {second_name} ({web_name})"


def get_currency_representation(amount: int):
    """Formats game currency as a string."""
    return f"${round(amount / 10, 1)}"


def print_simulated_gameweek_report(
        initial_squad: set, final_squad: set, 
        selected_squad_roles: set, substituted_squad_roles: set, 
        initial_budget: int, final_budget: int, 
        initial_squad_selling_prices: pd.Series, 
        final_squad_selling_prices: pd.Series,
        final_squad_purchase_prices: pd.Series,
        gameweek: int, 
        player_gameweek_points: pd.Series, 
        gameweek_points: int, 
        total_points: int,
        positions: pd.Series, 
        elements: pd.DataFrame
    ):
    """Print a detailed summary of activity in a simulated gameweek."""

    position_names = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}

    print()
    print("-----------------------")
    print(f"Gameweek {gameweek}: {gameweek_points} points")
    print("-----------------------")

    # Print the starting XI, including substitutions, names, and captaincy
    print("\nStarting XI")
    for player in sorted(substituted_squad_roles['starting_xi'], key=lambda player: positions.loc[player]):
    
        if player not in selected_squad_roles['starting_xi']:
            print("->", end=" ")
        else:
            print("  ", end=" ")

        if player == substituted_squad_roles['captain']:
            print("(C)", end=" ")
        elif player == substituted_squad_roles['vice_captain']:
            print("(V)", end=" ")
        else:
            print("   ", end=" ")

        print(f"{position_names[positions.loc[player]]}", end=" ")
        print(f"{get_full_name(player, elements)}", end=" ")
        print(f"[{sum_player_points([player], player_gameweek_points)} pts]", end=" ")
        print()

    # Print the reserve players
    print("\nReserves")
    for player in [substituted_squad_roles['reserve_gkp'], *substituted_squad_roles['reserve_out']]:

        if player in selected_squad_roles['starting_xi']:
            print("<-", end=" ")
        else:
            print("  ", end=" ")

        if player == selected_squad_roles['captain']:
            print("(*C)", end=" ")
        elif player == selected_squad_roles['vice_captain']:
            print("(*V)", end=" ")
        else:
            print("    ", end=" ")

        print(f"{position_names[positions.loc[player]]}", end=" ")
        print(f"{get_full_name(player, elements)}", end=" ")
        print(f"[{sum_player_points([player], player_gameweek_points)} pts]", end=" ")
        print()

    # Print transfer information
    print("\nTransfers")
    for player in (final_squad - initial_squad):
        print(f"-> {get_full_name(player, elements)} ({get_currency_representation(final_squad_purchase_prices.loc[player])})")
    for player in (initial_squad - final_squad):
        print(f"<- {get_full_name(player, elements)} ({get_currency_representation(initial_squad_selling_prices.loc[player])})")

    # Print budget and squad value
    value = final_squad_selling_prices.loc[list(final_squad)].sum()
    print(f"\nBank: {get_currency_representation(final_budget)}")
    print(f"Value: {get_currency_representation(value)}")

    # Print total points
    print(f"\nTotal points: {total_points}")


def get_initial_team_and_budget(season: str):

    if season == '2021-22':
        initial_squad = {389, 237, 220, 275, 233, 359, 485, 196, 43, 177, 40, 376, 14, 62, 348}
        initial_budget = 0
    elif season == '2022-23':
        initial_squad = {81, 100, 285, 306, 139, 448, 14, 283, 446, 374, 486, 381, 80, 28, 237}
        initial_budget = 45
    elif season == '2023-24':
        initial_squad = {352, 616, 206, 398, 209, 6, 501, 294, 303, 415, 297, 597, 368, 31, 278}
        initial_budget = 35

    return initial_squad, initial_budget


def run_simulation(season: str, log=False, use_cache=True) -> int:
    """Simulate a season of FPL and return the total points scored."""
    
    # Set up initial conditions and load results
    first_gameweek = 1
    last_gameweek = 38
    wildcard_gameweeks = (11, 26)
    initial_squad, initial_budget = get_initial_team_and_budget(season)
    current_squad, current_budget = initial_squad, initial_budget
    purchase_prices = load_simulation_purchase_prices(season, current_squad, first_gameweek)
    true_results = load_simulation_true_results(season, use_cache=use_cache)
    true_total_points = true_results['total_points']
    true_minutes = true_results['minutes']


    total_points = 0
    for next_gameweek in range(first_gameweek, last_gameweek + 1):

        # Skip GW7 in the 2022-23 season
        if (season == '2022-23') and (next_gameweek == 7):
            continue

        # Load player data for the next gameweek
        gameweek_elements = load_simulation_bootstrap_elements(season, next_gameweek)
        now_costs = gameweek_elements['now_cost']
        positions = gameweek_elements['element_type']
        selling_prices = get_selling_prices(current_squad, purchase_prices, now_costs)

        # Make, aggregate and process predictions
        features = load_simulation_features(season, next_gameweek, use_cache=use_cache)
        model_path = f"models/ensemble/excluded-{season}.pkl"
        columns_path = f"models/ensemble/columns.json"
        predictions = make_predictions(features, model_path, columns_path)
        gameweek_predictions = group_predictions_by_gameweek(predictions)
        gameweek_predictions = weight_gameweek_predictions_by_availability(gameweek_predictions, gameweek_elements, next_gameweek)

        # Check which gameweeks to optimize for.
        future_gameweeks = get_future_gameweeks(next_gameweek, last_gameweek, wildcard_gameweeks)
        if (season == '2022-23') and (7 in future_gameweeks):
            future_gameweeks.remove(7)

        # Make transfers and substitutions
        if (next_gameweek == 1) or (next_gameweek in wildcard_gameweeks):
            best_squad = run_greedy_optimization(
                current_squad, current_budget, future_gameweeks, 
                now_costs, gameweek_elements, selling_prices, gameweek_predictions,
            )
        else:
            best_squad = make_best_transfer(
                current_squad, future_gameweeks, current_budget, 
                gameweek_elements, selling_prices, gameweek_predictions,
            )

        # Update budgets and prices
        best_squad_budget = calculate_budget(
            current_squad, best_squad, current_budget, selling_prices, now_costs
        )
        updated_purchase_prices = update_purchase_prices(
            purchase_prices, now_costs, current_squad, best_squad
        )
        updated_selling_prices = update_selling_prices(
            selling_prices, now_costs, current_squad, best_squad
        )

        # Make automatic substitutions
        best_squad_roles = suggest_squad_roles(
            best_squad, positions, gameweek_predictions.loc[:, next_gameweek]
        )
        substituted_roles = make_automatic_substitutions(
            best_squad_roles, true_minutes.loc[:, next_gameweek], positions
        )

        # Calculate points
        gameweek_points = calculate_points(
            roles=substituted_roles,
            total_points=true_total_points.loc[:, next_gameweek],
            captain_multiplier=2,
            starting_xi_multiplier=1,
            reserve_gkp_multiplier=0,
            reserve_out_multiplier=0
        )
        total_points += gameweek_points

        # Print the gameweek report
        if log:
            print_simulated_gameweek_report(
                initial_squad=current_squad, final_squad=best_squad,
                selected_squad_roles=best_squad_roles, substituted_squad_roles=substituted_roles,
                initial_budget=initial_budget, final_budget=best_squad_budget,
                initial_squad_selling_prices=selling_prices, final_squad_selling_prices=updated_selling_prices,
                final_squad_purchase_prices=updated_purchase_prices,
                gameweek=next_gameweek, player_gameweek_points=true_total_points.loc[:, next_gameweek], 
                gameweek_points=gameweek_points, total_points=total_points,
                positions=positions, elements=gameweek_elements
            )

        # Carry out updates
        current_squad = best_squad
        current_budget = best_squad_budget
        purchase_prices = updated_purchase_prices
        selling_prices = updated_selling_prices

    return total_points