from types import MappingProxyType

from game.rules import CAPTAIN_MULTIPLIER

# Keep default parameters immutable
default_parameters = MappingProxyType(
    {
        "optimization_window_size": 6,
        "round_decay": 0.84,
        "starting_xi_multiplier": 1.0,
        "captain_multiplier": CAPTAIN_MULTIPLIER,
        "vice_captain_multiplier": 1.1,
        "reserve_gkp_multiplier": 0.03,
        "reserve_out_1_multiplier": 0.21,
        "reserve_out_2_multiplier": 0.06,
        "reserve_out_3_multiplier": 0.002,
        "budget_value": 0.008,
        "free_transfer_value": 1.0,
        "transfer_cost_multiplier": 1.0,
        "filter_percentile": 0.5,
    }
)


def get_parameters(overrides: dict = None) -> MappingProxyType:
    """Get optimization parameters, applying any overrides."""
    params = dict(default_parameters)
    if overrides:
        params.update(overrides)
    return MappingProxyType(params)
