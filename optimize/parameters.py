import numpy as np


parameters = {
    'squad_evaluation_round_factor': 0.45666334171014294,
    'captain_multiplier': 2,
    'starting_xi_multiplier': 1,
    'reserve_gkp_multiplier': 0.768046237286912,
    'reserve_out_multiplier': 0.9513395545369758 ** np.arange(1, 4),
    'future_gameweeks_evaluated': 8,
    'budget_importance': 0.00028173625403722804,
}


def get_parameter(name: str):
    return parameters[name]


def set_parameter(name: str, value):
    parameters[name] = value