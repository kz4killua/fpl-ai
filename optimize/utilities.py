import pandas as pd
import numpy as np

from predictions import sum_gameweek_predictions
from optimize.parameters import CAPTAIN_MULTIPLIER, RESERVE_GKP_MULTIPLIER, RESERVE_OUT_MULTIPLIER, STARTING_XI_MULTIPLER, SQUAD_EVALUATION_ROUND_FACTOR

GKP = 1
DEF = 2
MID = 3
FWD = 4


def suggest_squad_roles(squad: set, gameweek: int, positions: pd.Series, gameweek_predictions: pd.Series) -> dict:
    """
    Suggests captaincy and starting XI choices for a squad combination.
    """

    # Sort the squad in descending order of points
    squad = sorted(
        squad, 
        key=lambda player: sum_gameweek_predictions([player], gameweek, gameweek_predictions), 
        reverse=True
    )

    captain, vice_captain = squad[0], squad[1]

    starting_gkp = None
    reserve_gkp = None

    starting_defs = list()
    starting_mids = list()
    starting_fwds = list()

    reserve_out = list()

    for player in squad:

        # Count how many non required starters we have
        non_required = 0
        non_required += max(len(starting_defs) - 3, 0)
        non_required += max(len(starting_mids) - 0, 0)
        non_required += max(len(starting_fwds) - 1, 0)

        if positions[player] == GKP:
            if starting_gkp is None:
                starting_gkp = player
            else:
                reserve_gkp = player
            
        elif positions[player] == DEF:
            if len(starting_defs) < 3 or non_required < 6:
                starting_defs.append(player)
            else:
                reserve_out.append(player)

        elif positions[player] == MID:
            if len(starting_mids) < 0 or non_required < 6:
                starting_mids.append(player)
            else:
                reserve_out.append(player)
        
        else:
            if len(starting_fwds) < 1 or non_required < 6:
                starting_fwds.append(player)
            else:
                reserve_out.append(player)

    return {
        'captain': captain,
        'vice_captain': vice_captain,
        'starting_xi': [starting_gkp, *starting_defs, *starting_mids, *starting_fwds],
        'reserve_out': reserve_out,
        'reserve_gkp': reserve_gkp
    }


def calculate_points(roles, gameweek_predictions, gameweek, captain_multiplier=CAPTAIN_MULTIPLIER, starting_xi_multiplier=STARTING_XI_MULTIPLER, reserve_gkp_multiplier=RESERVE_GKP_MULTIPLIER, reserve_out_multiplier=RESERVE_OUT_MULTIPLIER):
    """
    Calculates the predicted points haul for a single gameweek.
    """

    points = 0

    # Sum up predictions for each position
    points += sum_gameweek_predictions(
        players=[roles['captain']], 
        gameweek=gameweek, 
        gameweek_predictions=gameweek_predictions,
        weights=captain_multiplier
    )
    points += sum_gameweek_predictions(
        players=list(set(roles['starting_xi']) - {roles['captain']}),
        gameweek=gameweek,
        gameweek_predictions=gameweek_predictions,
        weights=starting_xi_multiplier
    )
    points += sum_gameweek_predictions(
        players=[roles['reserve_gkp']],
        gameweek=gameweek,
        gameweek_predictions=gameweek_predictions,
        weights=reserve_gkp_multiplier
    )
    points += sum_gameweek_predictions(
        players=roles['reserve_out'],
        gameweek=gameweek,
        gameweek_predictions=gameweek_predictions,
        weights=reserve_out_multiplier
    )

    return points


def evaluate_squad(squad, positions, gameweeks, gameweek_predictions, squad_evaluation_round_factor=SQUAD_EVALUATION_ROUND_FACTOR, captain_multiplier=CAPTAIN_MULTIPLIER, starting_xi_multiplier=STARTING_XI_MULTIPLER, reserve_gkp_multiplier=RESERVE_GKP_MULTIPLIER, reserve_out_multiplier=RESERVE_OUT_MULTIPLIER):
    """
    Returns a score representing the 'goodness' of a squad for upcoming 'gameweeks'.
    """

    scores = []

    # Sum up the predicted points haul for each gameweek.
    for gameweek in gameweeks:
        roles = suggest_squad_roles(
            squad, gameweek, positions, gameweek_predictions
        )
        scores.append(
            calculate_points(
                roles, gameweek_predictions, gameweek,
                captain_multiplier=captain_multiplier,
                starting_xi_multiplier=starting_xi_multiplier,
                reserve_gkp_multiplier=reserve_gkp_multiplier,
                reserve_out_multiplier=reserve_out_multiplier
            )
        )

    # Apply weights to the score for each gameweek. 
    weights = (squad_evaluation_round_factor ** np.arange(len(scores)))
    weights /= weights.sum()
    scores *= weights

    return scores.sum()


def get_valid_transfers(squad: set, player_out: int, elements: pd.DataFrame, selling_prices: pd.Series, budget: int) -> set:
    """
    Returns all players that can be legally transferred in.
    """

    new_squad = squad - {player_out}
    player_out_position = elements.loc[player_out, 'element_type']
    
    # Calculate the budget left after selling 'player_out'
    budget_left = budget + selling_prices[player_out]

    # Check which teams already have 3 players in the squad
    team_counts = elements.loc[list(new_squad), 'team'].value_counts()
    invalid_teams = set(team_counts[team_counts >= 3].index)

    # Filter the 'elements' dataframe to contain only valid players
    valid_transfers = elements[
        (elements['element_type'] == player_out_position)
        & (elements['now_cost'] <= budget_left)
        & (elements['chance_of_playing_next_round'] == 100)
        & ~(elements['team'].isin(invalid_teams))
        & ~(elements['id'].isin(new_squad))
    ]

    return set(valid_transfers['id']) | {player_out}


def make_best_transfer(squad: set, gameweeks: list, budget: int, elements: pd.DataFrame, selling_prices: pd.Series, gameweek_predictions: pd.DataFrame) -> set:
    """
    Find the best single transfer that can be made.
    """

    elements.set_index('id', drop=False, inplace=True)
    positions = elements['element_type']
    
    best_squad = squad
    best_squad_evaluation = evaluate_squad(squad, positions, gameweeks, gameweek_predictions)

    # Try out all valid transfers
    for player_out in squad:
        for player_in in get_valid_transfers(squad, player_out, elements, selling_prices, budget):
            
            # Evaluate the new squad
            new_squad = squad - {player_out} | {player_in}
            new_squad_evaluation = evaluate_squad(new_squad, positions, gameweeks, gameweek_predictions)

            # Keep only the best squad
            if new_squad_evaluation > best_squad_evaluation:
                best_squad = new_squad
                best_squad_evaluation = new_squad_evaluation

    return best_squad