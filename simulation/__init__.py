import json

import pandas as pd
import numpy as np

from datautil.pipeline import load_players_and_teams
from datautil.utilities import get_previous_seasons
from datautil.constants import LOCAL_DATA_PATH
from features.features import engineer_features
from optimize.utilities import make_best_transfer, suggest_squad_roles, get_future_gameweeks, calculate_points, calculate_budget, sum_player_points
from predictions import make_predictions, group_predictions_by_gameweek, weight_gameweek_predictions_by_availability
from simulation.utilities import make_automatic_substitutions, get_selling_prices, update_purchase_prices, update_selling_prices
from optimize.greedy import run_greedy_optimization


def get_full_name(player_id: int, elements: pd.DataFrame):
    """
    Returns a player's full name, formatted as `first second (web)`.
    """
    first_name = elements.loc[player_id, 'first_name']
    second_name = elements.loc[player_id, 'second_name']
    web_name = elements.loc[player_id, 'web_name']
    return f"{first_name} {second_name} ({web_name})"


def get_currency_representation(amount: int):
    """
    Formats game currency as a string.
    """
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
    """
    Print a detailed summary of activity in a simulated gameweek.
    """

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


def get_player_costs(season: str, squad: set, gameweek: int):
    """
    Returns the `now_cost` of all players in the squad.
    """
    elements = load_gameweek_elements(season, gameweek)
    purchase_prices = elements['now_cost'].loc[list(squad)]
    return purchase_prices


def load_gameweek_elements(season: str, next_gameweek: int):
    """
    Load player data for the next gameweek.
    """

    with open(LOCAL_DATA_PATH / f"api/{season}/bootstrap/after_gameweek_{next_gameweek-1}.json") as f:
        bootstrap = json.load(f)
    elements = pd.DataFrame(bootstrap['elements'])
    elements.set_index('id', inplace=True, drop=False)
    elements['chance_of_playing_next_round'].fillna(100, inplace=True)

    return elements


def _run_simulation(
        season, initial_squad, initial_budget, 
        predictions, true_minutes, true_total_points,
        wildcard_gameweeks=[11, 26], first_gameweek=1, last_gameweek=38, log=False
    ):
    
    current_squad = initial_squad.copy()
    current_budget = initial_budget
    purchase_prices = get_player_costs(season, current_squad, first_gameweek)

    total_points = 0
    for next_gameweek in range(first_gameweek, last_gameweek + 1):

        # Skip GW7 in the 2022-23 season
        if (season == '2022-23') and (next_gameweek == 7):
            continue

        # Load player data for the next gameweek
        gameweek_elements = load_gameweek_elements(season, next_gameweek)
        now_costs = gameweek_elements['now_cost']
        positions = gameweek_elements['element_type']
        selling_prices = get_selling_prices(current_squad, purchase_prices, now_costs)

        # Aggregate and pre-process predictions
        gameweek_predictions = group_predictions_by_gameweek(predictions)
        gameweek_predictions = gameweek_predictions.loc[gameweek_elements['id'].values, :]
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


def run_simulation(season: str, log=False, use_cache=True):
    """
    Simulates a season of FPL and returns the total points scored.
    """
    
    fixtures = pd.read_csv(f"data/api/{season}/fixtures.csv")
    fixtures['kickoff_time'] = pd.to_datetime(fixtures['kickoff_time'])

    # Load predictions and features
    if use_cache:
        local_players = pd.read_pickle(f'cache/local_players-{season}.pkl')
        features = pd.read_pickle(f'cache/features-{season}.pkl')
        predictions = pd.read_pickle(f'cache/predictions-{season}.pkl')
    else:
        previous_seasons = get_previous_seasons(season)
        local_players, local_teams = load_players_and_teams(previous_seasons)
        features, columns = engineer_features(local_players, local_teams)
        model_path = f"models/ensemble/excluded-{season}.pkl"
        columns_path = f"models/ensemble/columns.json"
        predictions = make_predictions(features, model_path, columns_path)

    # Get true points and minutes
    season_players = local_players[local_players['season'] == season]
    true_total_points = season_players[
        ['element', 'round', 'total_points']
    ].groupby(['element', 'round']).sum()['total_points']
    true_minutes = season_players[
        ['element', 'round', 'minutes']
    ].groupby(['element', 'round']).sum()['minutes']

    initial_squad, initial_budget = get_initial_team_and_budget(season)
    total_points = _run_simulation(
        season, initial_squad, initial_budget, predictions, 
        true_minutes, true_total_points, log=log
    )

    return total_points