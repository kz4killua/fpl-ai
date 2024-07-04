import pandas as pd
import numpy as np

from predictions import sum_player_points
from optimize.parameters import CAPTAIN_MULTIPLIER, RESERVE_GKP_MULTIPLIER, RESERVE_OUT_MULTIPLIER, STARTING_XI_MULTIPLER, SQUAD_EVALUATION_ROUND_FACTOR, FUTURE_GAMEWEEKS_EVALUATED

GKP = 1
DEF = 2
MID = 3
FWD = 4


def suggest_squad_roles(squad: set, positions: pd.Series, total_points: pd.Series) -> dict:
    """
    Suggests captaincy and starting XI choices for a squad combination.
    """

    # Sort the squad in descending order of total points for the gameweek
    squad = sorted(
        squad, 
        key=lambda player: sum_player_points([player], total_points), 
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


def calculate_points(roles: dict, total_points: pd.Series, captain_multiplier, starting_xi_multiplier, reserve_gkp_multiplier, reserve_out_multiplier):
    """
    Calculates the points haul for a single gameweek.
    """

    points = 0

    # Sum up points for each position
    points += sum_player_points(
        players=[roles['captain']],
        total_points=total_points,
        weights=captain_multiplier
    )
    points += sum_player_points(
        players=list(set(roles['starting_xi']) - {roles['captain']}),
        total_points=total_points,
        weights=starting_xi_multiplier
    )
    points += sum_player_points(
        players=[roles['reserve_gkp']],
        total_points=total_points,
        weights=reserve_gkp_multiplier
    )
    points += sum_player_points(
        players=roles['reserve_out'],
        total_points=total_points,
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
        total_points = gameweek_predictions.loc[:, gameweek]
        roles = suggest_squad_roles(
            squad, positions, total_points
        )
        scores.append(
            calculate_points(
                roles=roles, 
                total_points=total_points,
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

    assert elements.index.name == "id"
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


def get_future_gameweeks(next_gameweek, last_gameweek=38, wildcard_gameweeks=[], future_gameweeks_evaluated=FUTURE_GAMEWEEKS_EVALUATED):
    """List out the gameweeks to optimize over."""

    future_gameweeks = list(range(next_gameweek, min(last_gameweek + 1, next_gameweek + future_gameweeks_evaluated)))

    # Do not evaluate past the next wildcard gameweek
    if wildcard_gameweeks:
        for gameweek in sorted(wildcard_gameweeks):
            if gameweek > next_gameweek:
                future_gameweeks = [x for x in future_gameweeks if x < gameweek]
                break

    return future_gameweeks


def calculate_budget(initial_squad: set, final_squad: set, initial_budget: int, selling_prices: pd.Series, now_costs: pd.Series) -> int:
    """Calculate the budget change after moving from an initial to a final squad."""    
    transfers_in = final_squad - initial_squad
    transfers_out = initial_squad - final_squad
    final_budget = initial_budget + selling_prices.loc[list(transfers_out)].sum() - now_costs.loc[list(transfers_in)].sum()
    return final_budget