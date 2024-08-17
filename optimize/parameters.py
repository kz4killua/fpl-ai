import numpy as np


parameters = {
    'squad_evaluation_round_factor': 0.7915879292906937,
    'captain_multiplier': 2,
    'starting_xi_multiplier': 1,
    'reserve_gkp_multiplier': 0.26780126530565046,
    'reserve_out_multiplier': 0.3749016858132089 ** np.arange(1, 4),
    'future_gameweeks_evaluated': 4,
    'budget_importance': 3.490974459751069e-07,
}


def get_parameter(name: str):
    return parameters[name]


def set_parameter(name: str, value):
    if name not in parameters:
        raise ValueError(f"Parameter '{name}' not found.")
    parameters[name] = value