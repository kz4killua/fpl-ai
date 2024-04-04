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
        player: minutes.get(player, np.int64(0)).sum() for player in minutes.keys().unique()
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