import math
import time
import random

import numpy as np

TRANSFER_COST = 4
GKP, DEF, MID, FWD = 1, 2, 3, 4

ANNEALING_TIME = 600
MAX_TEMPERATURE = 1

MAX_UPPER_GAMEWEEKS = 5
NEXT_GAMEWEEK_WEIGHT = 1
UPPER_GAMEWEEK_WEIGHT = 0.5

CAPTAIN_MULTIPLIER = 2
STARTING_XI_MULTIPLIER = 1
RESERVE_GKP_MULTIPLIER = 0.039
RESERVE_OUT_MULTIPLIER = np.array([0.4425, 0.195, 0.0775])

TRANSFER_CONFIDENCE = 0.5


def suggest_squad_roles(squad, element_types, predictions):
    """Suggests captain and starting XI choices for a squad."""

    # Sort the squad in descending order of points
    players = sorted(squad, key=lambda player: predictions[player], reverse=True)

    # The best two players should be captain and vice captain
    captain, vice_captain = players[0], players[1]

    starting_gkp = None
    reserve_gkp = None

    starting_defs = list()
    starting_mids = list()
    starting_fwds = list()

    reserve_out = list()

    for player in players:

        if element_types[player] == GKP:
            if starting_gkp is None:
                starting_gkp = player
            else:
                reserve_gkp = player
            continue

        # Count how many non required starters we have
        non_required = 0
        non_required += max(len(starting_defs) - 3, 0)
        non_required += max(len(starting_mids) - 0, 0)
        non_required += max(len(starting_fwds) - 1, 0)

        # If we have room for non-required players, add to the starting XI
        if element_types[player] == DEF:
            if len(starting_defs) < 3 or non_required < 6:
                starting_defs.append(player)
            else:
                reserve_out.append(player)

        elif element_types[player] == MID:
            if len(starting_mids) < 0 or non_required < 6:
                starting_mids.append(player)
            else:
                reserve_out.append(player)
        else:
            if len(starting_fwds) < 1 or non_required < 6:
                starting_fwds.append(player)
            else:
                reserve_out.append(player)

    # Package
    squad_roles = {
        'captain': captain,
        'vice_captain': vice_captain,
        'starting_xi': [starting_gkp, *starting_defs, *starting_mids, *starting_fwds],
        'reserve_out': reserve_out,
        'reserve_gkp': reserve_gkp
    }

    return squad_roles


def calculate_total_transfer_cost(squad, initial_squad, free_transfers):
    """Calculates the points hit that will be taken to switch from 
    an initial squad to a given squad."""
    # Count the number of players transferred in.
    transfers_made = len(squad - initial_squad)
    # Check the total cost of transfers.
    total_transfer_cost = max((transfers_made - free_transfers), 0) * TRANSFER_COST
    return total_transfer_cost


def evaluate_squad(squad, element_types, next_gameweek_predictions, upper_gameweek_predictions, initial_squad, free_transfers, squad_evaluations):
    """Returns the 'goodness' of a squad for both 
    the next gameweek and future gameweeks. """

    # Check if we have evaluated this squad before.
    if frozenset(squad) in squad_evaluations:
        return squad_evaluations[frozenset(squad)]

    # Estimate the number of points the squad will make in the next gameweek.
    next_gw_score = next_gameweek_score(
        squad, element_types, next_gameweek_predictions, initial_squad, free_transfers)
    
    # Estimate the number of points that can be gotten in future gameweeks.
    upper_gw_score = upper_gameweek_score(squad, element_types, upper_gameweek_predictions)

    # Calculate a score based on next week's estimate and that of future weeks.
    score = (next_gw_score * NEXT_GAMEWEEK_WEIGHT) + (upper_gw_score * UPPER_GAMEWEEK_WEIGHT)

    # Store evaluations.
    squad_evaluations[frozenset(squad)] = score
        
    return score


def gameweek_score(squad, element_types, predictions):
    """Calculates the number of points predicted for a squad in a gameweek."""
    score = 0

    # Get the best XI.
    squad_roles = suggest_squad_roles(squad, element_types, predictions)

    # Score the captain.
    score += predictions[squad_roles['captain']] * CAPTAIN_MULTIPLIER
    # Score other starting players.
    score += predictions[list(set(squad_roles['starting_xi']) - {squad_roles['captain']})].sum() * STARTING_XI_MULTIPLIER
    # Score the reserve GKP.
    score += predictions[squad_roles['reserve_gkp']] * RESERVE_GKP_MULTIPLIER
    # Score the reserve outfield players.
    score += (predictions[squad_roles['reserve_out']] * RESERVE_OUT_MULTIPLIER).sum()

    return score


def next_gameweek_score(squad, element_types, next_gameweek_predictions, initial_squad, free_transfers):
    """Calculates the number of points predicted for a given squad in the next gameweek."""
    score = gameweek_score(squad, element_types, next_gameweek_predictions)
    # Calculate total transfer cost.
    total_transfer_cost = calculate_total_transfer_cost(squad, initial_squad, free_transfers)
    total_transfer_cost /= TRANSFER_CONFIDENCE
    # Subtract from the score.
    score -= total_transfer_cost
    return score   


def upper_gameweek_score(squad, element_types, upper_gameweek_predictions):
    """Calculates the number of points predicted for a squad over future gameweeks."""
    score = sum(
        gameweek_score(squad, element_types, gameweek_predictions)
        for gameweek_predictions in upper_gameweek_predictions
    )
    return score


def calculate_budget(new_squad, initial_squad, initial_budget, current_costs, selling_prices):
    """Calculates how much money is left in the budget."""
    # Get players added and removed
    transfers_in = new_squad - initial_squad
    transfers_out = initial_squad - new_squad
    # Calculate income and expenses
    expenses = current_costs[list(transfers_in)].sum()
    income = selling_prices[list(transfers_out)].sum()
    # Calculate budget remaining
    funds_left = initial_budget - expenses + income

    return funds_left


def make_random_transfer(squad, initial_squad, selling_prices, elements, initial_budget): 
    """Randomly switches out one player in the squad."""

    new_squad = squad.copy()

    # Remove a random player from the squad
    player_out = random.choice(list(new_squad))
    new_squad.remove(player_out)

    # Check the player's position
    player_out_position = elements.loc[player_out, 'element_type']
    
    # Check which teams have up to 3 players in our squad
    team_counts = elements.loc[list(new_squad), 'team'].value_counts()
    flagged_teams = set(team_counts[team_counts >= 3].index)

    # Calculate how much money we have left
    funds_left = calculate_budget(new_squad, initial_squad, initial_budget, elements['now_cost'], selling_prices)

    # Filter valid players
    filtered_players = elements[
        (elements['element_type'] == player_out_position) &
        (elements['now_cost'] <= funds_left) &
        (elements['chance_of_playing_next_round'] == 100) &
        ~(elements['team'].isin(flagged_teams)) &
        ~(elements['id'].isin(new_squad))
    ]

    # Add a random player to the squad
    player_in = random.choice(list(filtered_players['id']))
    new_squad.add(player_in)

    return new_squad


def get_valid_transfers_in(initial_squad, player_out, elements, initial_budget, selling_prices):
    """Returns all players that can be legally transferred in."""

    position = elements.loc[player_out, 'element_type']
    new_squad = initial_squad - {player_out}
    funds_left = calculate_budget(new_squad, initial_squad, initial_budget, elements['now_cost'], selling_prices)

    # Check which teams have up to 3 players in the squad
    team_counts = elements.loc[list(initial_squad), 'team'].value_counts()
    flagged_teams = set(team_counts[team_counts >= 3].index)

    # Filter valid players 
    filtered_elements = elements[
        (elements['element_type'] == position) &
        (elements['now_cost'] <= funds_left) &
        (elements['chance_of_playing_next_round'] == 100) &
        ~(elements['team'].isin(flagged_teams)) &
        ~(elements['id'].isin(new_squad))
    ]
    valid_players = list(filtered_elements['id'])

    return valid_players


def best_transfer(initial_squad, initial_budget, elements, selling_prices, next_gameweek_predictions, upper_gameweek_predictions, free_transfers):
    """Returns the best single transfer that can be made."""

    # Remember to index elements by ID without dropping
    elements = elements.set_index('id', drop=False)

    best_squad = None
    best_squad_evaluation = float('-inf')

    for player_out in initial_squad:
        
        valid_transfers = get_valid_transfers_in(initial_squad, player_out, elements, initial_budget, selling_prices)

        # Try all valid transfers
        for player_in in valid_transfers:

            new_squad = initial_squad - {player_out} | {player_in}
            squad_evaluation = evaluate_squad(new_squad, elements['element_type'], next_gameweek_predictions, upper_gameweek_predictions, initial_squad, free_transfers, {})

            if squad_evaluation > best_squad_evaluation:
                best_squad_evaluation = squad_evaluation
                best_squad = new_squad

    return best_squad


def get_temperature(elapsed):
    """Returns the temperature at a particular time-step."""
    return ((ANNEALING_TIME - elapsed) / ANNEALING_TIME) * MAX_TEMPERATURE


def simulated_annealing(initial_squad, selling_prices, free_transfers, initial_budget_remaining, next_gameweek_predictions, upper_gameweek_predictions, elements):
    """Uses a simulated annealing algorithm to find the best transfers to make."""

    # We will access players by their ids
    elements = elements.set_index('id', drop=False)

    # Start with the initial squad.
    current = initial_squad.copy()

    squad_evaluations = dict()

    # We will keep track of the best squad and score.
    best_squad = current
    best_score = evaluate_squad(
        current, elements['element_type'], 
        next_gameweek_predictions, upper_gameweek_predictions, 
        initial_squad, free_transfers, squad_evaluations)

    # Record the time when annealing started
    start = time.time()

    while True:

        # Check how many seconds have passed since the start
        elapsed = time.time() - start
        # Quit the program if we are out of time
        if elapsed >= ANNEALING_TIME:
            break
        # Calculate the temperature at the current time step
        temperature = get_temperature(elapsed)

        neighbour = make_random_transfer(
            current, initial_squad, selling_prices, elements, initial_budget_remaining)

        # How much better is the neighbour than the current state?
        neighbour_score = evaluate_squad(neighbour, elements['element_type'], next_gameweek_predictions, upper_gameweek_predictions, initial_squad, free_transfers, squad_evaluations)
        current_score = evaluate_squad(current, elements['element_type'], next_gameweek_predictions, upper_gameweek_predictions, initial_squad, free_transfers, squad_evaluations)
        delta_e = neighbour_score - current_score

        # Keep track of the best squad.
        if neighbour_score > best_score:
            best_squad = neighbour
            best_score = neighbour_score

        # If neighbour is better than current, set current = neighbour
        if delta_e > 0:
            current = neighbour

        # Else, with probability e ** delta_e / temperature, set current = neighbour
        else:
            if temperature == 0:
                probability = 0
            else:
                probability = math.e ** (delta_e / temperature)

            if probability > random.random():
                current = neighbour

    return best_squad


if __name__ == '__main__':
    pass