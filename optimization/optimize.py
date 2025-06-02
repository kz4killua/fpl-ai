from game.rules import (
    CAPTAIN_MULTIPLIER,
)
from optimization.parameters import (
    RESERVE_GKP_MULTIPLIER,
    RESERVE_OUT_1_MULTIPLIER,
    RESERVE_OUT_2_MULTIPLIER,
    RESERVE_OUT_3_MULTIPLIER,
    ROUND_DECAY,
    STARTING_XI_MULTIPLIER,
    VICE_CAPTAIN_MULTIPLIER,
)
from optimization.solver import solve


def optimize_squad(
    initial_squad: list[int],
    initial_budget: int,
    initial_free_transfers: int,
    now_costs: dict[int, int],
    selling_prices: dict[int, int],
    upcoming_gameweeks: list[int],
    wildcard_gameweeks: list[int],
    total_points: dict[tuple[int, int], float],
    element_types: dict[int, int],
    teams: dict[int, int],
    # Optimization parameters
    round_decay: float = ROUND_DECAY,
    starting_xi_multiplier: float = STARTING_XI_MULTIPLIER,
    captain_multiplier: float = CAPTAIN_MULTIPLIER,
    vice_captain_multiplier: float = VICE_CAPTAIN_MULTIPLIER,
    reserve_gkp_multiplier: float = RESERVE_GKP_MULTIPLIER,
    reserve_out_1_multiplier: float = RESERVE_OUT_1_MULTIPLIER,
    reserve_out_2_multiplier: float = RESERVE_OUT_2_MULTIPLIER,
    reserve_out_3_multiplier: float = RESERVE_OUT_3_MULTIPLIER,
):
    """Runs optimization and returns the optimal squad roles for the next gameweek."""

    players = list(set(p for (p, _) in total_points))

    # Add gameweeks to player attributes
    teams = {(p, g): teams[p] for p in teams for g in upcoming_gameweeks}
    element_types = {
        (p, g): element_types[p] for p in element_types for g in upcoming_gameweeks
    }
    now_costs = {(p, g): now_costs[p] for p in now_costs for g in upcoming_gameweeks}
    selling_prices = {
        (p, g): selling_prices[p] for p in selling_prices for g in upcoming_gameweeks
    }

    # Add selling prices for players not in the squad
    for p in players:
        if p not in initial_squad:
            for g in upcoming_gameweeks:
                selling_prices[(p, g)] = now_costs[(p, g)]

    solution = solve(
        initial_squad=initial_squad,
        initial_budget=initial_budget,
        initial_free_transfers=initial_free_transfers,
        players=players,
        gameweeks=upcoming_gameweeks,
        wildcards=wildcard_gameweeks,
        total_points=total_points,
        teams=teams,
        element_types=element_types,
        now_costs=now_costs,
        selling_prices=selling_prices,
        round_decay=round_decay,
        starting_xi_multiplier=starting_xi_multiplier,
        captain_multiplier=captain_multiplier,
        vice_captain_multiplier=vice_captain_multiplier,
        reserve_gkp_multiplier=reserve_gkp_multiplier,
        reserve_out_1_multiplier=reserve_out_1_multiplier,
        reserve_out_2_multiplier=reserve_out_2_multiplier,
        reserve_out_3_multiplier=reserve_out_3_multiplier,
    )

    # Prepare roles for the next gameweek
    next_gameweek = upcoming_gameweeks[0]
    roles = {
        "starting_xi": solution[next_gameweek]["starting_xi"],
        "captain": solution[next_gameweek]["captain"][0],
        "vice_captain": solution[next_gameweek]["vice_captain"][0],
        "reserve_gkp": solution[next_gameweek]["reserve_gkp"][0],
        "reserve_out_1": solution[next_gameweek]["reserve_out_1"][0],
        "reserve_out_2": solution[next_gameweek]["reserve_out_2"][0],
        "reserve_out_3": solution[next_gameweek]["reserve_out_3"][0],
    }
    return roles
