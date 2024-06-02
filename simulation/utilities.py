import pandas as pd
import numpy as np
from copy import deepcopy

from optimize.utilities import GKP, DEF, MID, FWD, calculate_points


def make_automatic_substitutions(roles: set, minutes: pd.Series, positions: pd.Series) -> dict:
    """Returns squad roles after making automatic substitutions."""

    roles = deepcopy(roles)
    position_counts = positions.loc[list(roles['starting_xi'])].value_counts()
    squad_limits = {GKP: 1, DEF: 5, MID: 5, FWD: 3}
    minutes = {
        player: minutes.get(player, np.int64(0)).sum() for player in [*roles['starting_xi'], roles['reserve_gkp'], *roles['reserve_out']]
    }

    for i, player in enumerate(roles['starting_xi']):

        # Skip all players who played in the gameweek.
        if minutes[player] > 0:
            continue

        # Replace the starting GKP if necessary.
        elif positions.loc[player] == GKP and minutes[roles['reserve_gkp']] > 0: 
            roles['starting_xi'][i], roles['reserve_gkp'] = (
                roles['reserve_gkp'], roles['starting_xi'][i]
            )

        # Replace outfield players if necessary (and legal). 
        else:
            for j, reserve in enumerate(roles['reserve_out']):
                if minutes[reserve] > 0:
                    if position_counts[positions.loc[reserve]] < squad_limits[positions.loc[reserve]]:
                        roles['starting_xi'][i], roles['reserve_out'][j] = (
                            roles['reserve_out'][j], roles['starting_xi'][i]
                        )
                        position_counts[positions.loc[reserve]] += 1
                        break

    # Replace the captain if necessary
    if (minutes[roles['captain']] == 0) and (minutes[roles['vice_captain']] > 0):
        roles['captain'], roles['vice_captain'] = (
            roles['vice_captain'], roles['captain']
        )

    return roles


def calculate_selling_price(purchase_price: int, current_cost: int) -> int:
    """
    Calculates the selling price for a single player.
    """

    if current_cost <= purchase_price:
        return current_cost
    
    # Apply a 50% sell-on fee to any profits
    fee = round((current_cost - purchase_price) * 0.5)
    return current_cost - fee


def get_selling_prices(players: list, purchase_prices: pd.Series, now_costs: pd.Series) -> pd.Series:
    """
    Return a mapping of players to their selling prices.
    """

    selling_prices = pd.Series()

    for player in players:
        selling_prices.loc[player] = calculate_selling_price(
            purchase_price=purchase_prices.loc[player],
            current_cost=now_costs.loc[player]
        )

    return selling_prices


def update_purchase_prices(purchase_prices: pd.Series, now_costs: pd.Series, old_squad: set, new_squad: set) -> pd.Series:
    """
    Returns an updated record of purchase prices after a squad change.
    """

    purchase_prices = purchase_prices.copy()

    for player in new_squad - old_squad:
        purchase_prices.loc[player] = now_costs[player]
    for player in old_squad - new_squad:
        purchase_prices.drop(labels=player, inplace=True)

    return purchase_prices


def update_selling_prices(selling_prices: pd.Series, now_costs: pd.Series, old_squad: set, new_squad: set) -> pd.Series:
    """
    Returns an updated record of selling prices after a squad change.
    """

    selling_prices = selling_prices.copy()

    for player in new_squad - old_squad:
        selling_prices.loc[player] = now_costs[player]
    for player in old_squad - new_squad:
        selling_prices.drop(labels=player, inplace=True)

    return selling_prices