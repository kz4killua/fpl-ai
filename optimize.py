import math
import random

import pandas as pd

TEMPERATURE_FACTOR = 1

NEXT_GAMEWEEK_WEIGHT = 1
UPPER_GAMEWEEK_WEIGHT = 0.5

CAPTAIN_MULTIPLIER = 2
STARTING_XI_MULTIPLIER = 1
RESERVE_GKP_MULTIPLIER = 0.25
RESERVE_OUT_MULTIPLIER = 0.25

TRANSFER_CONFIDENCE = 0.5

squad_evaluations = {}

def expected_returns(player_or_squad, predictions):
    """Returns the expected returns of a player."""
    return predictions.loc[player_or_squad]

def suggest_squad_roles(squad, elements, predictions):

    starting_xi = list()
    reserve_out = list()

    # Get players and their positions.
    df = elements[elements['id'].isin(squad)][['id', 'element_type']].reset_index(drop=True)
    # Add predicted total points.
    df['expected_returns'] = expected_returns(df['id'], predictions).values

    # Sort the players by their expected returns.
    df = df.sort_values('expected_returns', ascending=False, ignore_index=True)

    # Get a list of all players in order of points.
    players = list(df['id'])
    # Split the squad by position.
    gkps = list(df[df['element_type'] == 1]['id'])
    defs = list(df[df['element_type'] == 2]['id'])
    mids = list(df[df['element_type'] == 3]['id'])
    fwds = list(df[df['element_type'] == 4]['id'])

    starting_defs = []
    starting_mids = []
    starting_fwds = []

    # The two best players should be captain and vice captain.
    captain = players[0]
    vice_captain = players[1]

    # The best GKP will start, the other will be on the bench.
    starting_gkp = gkps[0]
    reserve_gkp = gkps[1]

    # We must start at least 3 defenders.
    starting_defs.extend(defs[:3])
    # We must start at least 1 forward.
    starting_fwds.append(fwds[0])

    # The 6 best remaining outfield players will start.
    other_outfield_players = [
        element for element in players if 
        (element not in (set(starting_defs) | set(starting_mids) | set(starting_fwds))) and 
        (element not in gkps)][:6]
    # Add them to the right positions.
    starting_defs += list(set(other_outfield_players) & set(defs))
    starting_mids += list(set(other_outfield_players) & set(mids))
    starting_fwds += list(set(other_outfield_players) & set(fwds))

    starting_xi = [starting_gkp, *starting_defs, *starting_mids, *starting_fwds]

    # Get the reserve outfield players.
    reserve_out = list(set(squad) - set(starting_xi) - {reserve_gkp})

    # Sort the reserve outfield players in descending order of expected points.
    reserve_out = sorted(reserve_out, key=lambda player: expected_returns(player, predictions), reverse=True)

    # Package the information.
    output = {
        'captain': captain,
        'vice_captain': vice_captain,
        'starting_xi': starting_xi,
        'reserve_out': reserve_out,
        'reserve_gkp': reserve_gkp
    }

    return output


def calculate_total_transfer_cost(squad, initial_squad, transfer_cost, free_transfers):
    """Calculates the points hit that will be taken to switch from 
    an initial squad to a given squad."""
    # Count the number of players transferred in.
    transfers_made = len(squad - initial_squad)
    # Check the total cost of transfers.
    total_transfer_cost = max((transfers_made - free_transfers), 0) * transfer_cost
    return total_transfer_cost

def evaluate_squad(squad, elements, next_gameweek_predictions, upper_gameweek_predictions, number_of_upper_gameweeks, initial_squad, transfer_cost, free_transfers):
    """Returns the 'goodness' of a squad for both 
    the next gameweek and future gameweeks. """

    # Check if we have evaluated this squad before.
    if frozenset(squad) in squad_evaluations:
        return squad_evaluations[frozenset(squad)]

    # Estimate the number of points the squad will make in the next gameweek.
    next_gw_score = next_gameweek_score(
        squad, elements, next_gameweek_predictions, initial_squad, transfer_cost, free_transfers)
    
    # Estimate the number of points that can be gotten in future gameweeks.
    if number_of_upper_gameweeks == 0:
        upper_gw_score = 0
    else:
        upper_gw_score = upper_gameweek_score(squad, upper_gameweek_predictions)

    # Calculate a score based on next week's estimate and that of future weeks.
    score = (next_gw_score * NEXT_GAMEWEEK_WEIGHT) + (upper_gw_score * UPPER_GAMEWEEK_WEIGHT)

    # Store evaluations.
    squad_evaluations[frozenset(squad)] = score
        
    return score

def next_gameweek_score(squad, elements, next_gameweek_predictions, initial_squad, transfer_cost, free_transfers):
    """Calculates the number of points predicted for a given squad in the next gameweek."""
    
    score = 0

    # Get the best XI.
    squad_roles = suggest_squad_roles(squad, elements, next_gameweek_predictions)

    # Score the captain.
    score += expected_returns(squad_roles['captain'], next_gameweek_predictions) * CAPTAIN_MULTIPLIER  
    # Score other players in the starting XI.
    score += expected_returns(list(set(squad_roles['starting_xi']) - {squad_roles['captain']}), next_gameweek_predictions).sum() * STARTING_XI_MULTIPLIER
    # Score the reserve GKP.
    score += expected_returns(squad_roles['reserve_gkp'], next_gameweek_predictions) * RESERVE_GKP_MULTIPLIER
    # Score the reserve outfield players.
    score += expected_returns(list(squad_roles['reserve_out']), next_gameweek_predictions).sum() * RESERVE_OUT_MULTIPLIER

    # Calculate total transfer cost.
    total_transfer_cost = calculate_total_transfer_cost(squad, initial_squad, transfer_cost, free_transfers)
    total_transfer_cost /= TRANSFER_CONFIDENCE
    # Subtract from the score.
    score -= total_transfer_cost

    return score   

def upper_gameweek_score(squad, upper_gameweek_predictions):
    """Estimates the number of points expected for a squad over future gameweeks."""
    return expected_returns(list(squad), upper_gameweek_predictions).sum()
 

def make_random_transfer(squad, initial_squad, selling_prices, elements, initial_budget_remaining): 
    """Randomly switches out one player in the squad
    with another player in the same position from the 
    available players."""

    # Make a copy of the squad.
    neighbour = squad.copy()

    # Pick a random player to remove.
    player_out = random.choice(list(neighbour))
    # Check the player's position.
    player_out_position = elements.set_index('id').loc[player_out, 'element_type']
    # Remove the player from the squad.
    neighbour.remove(player_out)
    
    # Count the number of players in each team.
    team_counts = elements.set_index('id').loc[list(neighbour), 'team'].value_counts()
    # Check which teams are unavailable (already have more than 3 players).
    flagged_teams = set(team_counts[team_counts == 3].index)

    # Calculate how much money we have left.
    transfers_in = neighbour - initial_squad
    transfers_out = initial_squad - neighbour
    expenses = elements.set_index('id').loc[list(transfers_in), 'now_cost'].sum()
    income = selling_prices.set_index('element').loc[list(transfers_out), 'selling_price'].sum()
    funds_left = initial_budget_remaining - expenses + income

    # Filter valid players.
    filtered_players = elements[
        (elements['element_type'] == player_out_position) &
        (elements['now_cost'] <= funds_left) &
        (elements['chance_of_playing_next_round'] == 100) &
        ~(elements['team'].isin(flagged_teams)) &
        ~(elements['id'].isin(neighbour))
    ]

    # Pick a random player.
    player_in = random.choice(list(filtered_players['id']))
    # Add the player to the squad.
    neighbour.add(player_in)

    return neighbour

def get_temperature(t, max_t):
    """Returns the temperature at a particular time-step."""
    return ((max_t - t) / max_t) * TEMPERATURE_FACTOR

def simulated_annealing(initial_squad, selling_prices, transfer_cost, free_transfers, initial_budget_remaining, next_gameweek_predictions, upper_gameweek_predictions, number_of_upper_gameweeks, elements, max_t):
    """Uses a simulated annealing algorithm to find the best transfers to make."""

    # Start with the initial squad.
    current = initial_squad.copy()

    # We will keep track of the best squad and score.
    best_squad = current
    best_score = evaluate_squad(
        current, elements, 
        next_gameweek_predictions, upper_gameweek_predictions, number_of_upper_gameweeks, 
        initial_squad, transfer_cost, free_transfers)

    for t in range(1, max_t + 1):

        temperature = get_temperature(t, max_t)

        neighbour = make_random_transfer(
            current, initial_squad, selling_prices, elements, initial_budget_remaining)

        # How much better is the neighbour than the current state?
        neighbour_score = evaluate_squad(neighbour, elements, next_gameweek_predictions, upper_gameweek_predictions, number_of_upper_gameweeks, initial_squad, transfer_cost, free_transfers)
        current_score = evaluate_squad(current, elements, next_gameweek_predictions, upper_gameweek_predictions, number_of_upper_gameweeks, initial_squad, transfer_cost, free_transfers)
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