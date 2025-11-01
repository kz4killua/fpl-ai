from game.rules import CAPTAIN_MULTIPLIER
from game.utils import format_currency
from loaders.utils import print_table

from .parameters import (
    BUDGET_VALUE,
    FREE_TRANSFER_VALUE,
    RESERVE_GKP_MULTIPLIER,
    RESERVE_OUT_1_MULTIPLIER,
    RESERVE_OUT_2_MULTIPLIER,
    RESERVE_OUT_3_MULTIPLIER,
    ROUND_DECAY,
    STARTING_XI_MULTIPLIER,
    TRANSFER_COST_MULTIPLIER,
    VICE_CAPTAIN_MULTIPLIER,
)
from .solver import solve


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
    web_names: dict[int, str],
    parameters: dict[str, float] | None = None,
    log: bool = False,
):
    """Runs optimization and returns the optimal squad roles for the next gameweek."""

    # Prepare optimization data
    initial_squad = set(initial_squad)
    players = list(set(p for (p, _) in total_points))
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

    # Prepare optimization parameters
    parameters = parameters or {}
    round_decay = parameters.get("round_decay", ROUND_DECAY)
    starting_xi_multiplier = parameters.get(
        "starting_xi_multiplier", STARTING_XI_MULTIPLIER
    )
    captain_multiplier = parameters.get("captain_multiplier", CAPTAIN_MULTIPLIER)
    vice_captain_multiplier = parameters.get(
        "vice_captain_multiplier", VICE_CAPTAIN_MULTIPLIER
    )
    reserve_gkp_multiplier = parameters.get(
        "reserve_gkp_multiplier", RESERVE_GKP_MULTIPLIER
    )
    reserve_out_1_multiplier = parameters.get(
        "reserve_out_1_multiplier", RESERVE_OUT_1_MULTIPLIER
    )
    reserve_out_2_multiplier = parameters.get(
        "reserve_out_2_multiplier", RESERVE_OUT_2_MULTIPLIER
    )
    reserve_out_3_multiplier = parameters.get(
        "reserve_out_3_multiplier", RESERVE_OUT_3_MULTIPLIER
    )
    budget_value = parameters.get("budget_value", BUDGET_VALUE)
    free_transfer_value = parameters.get("free_transfer_value", FREE_TRANSFER_VALUE)
    transfer_cost_multiplier = parameters.get(
        "transfer_cost_multiplier", TRANSFER_COST_MULTIPLIER
    )

    # Solve the optimization problem
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
        budget_value=budget_value,
        free_transfer_value=free_transfer_value,
        transfer_cost_multiplier=transfer_cost_multiplier,
        log=log,
    )
    if log:
        print_optimization_solution(solution, web_names, element_types)

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


def print_optimization_solution(
    solution: dict, web_names: dict[int, str], element_types: dict[tuple[int, int], int]
):
    """Prints the solution in a readable format."""

    for gameweek in solution:
        purchases = solution[gameweek]["purchases"]
        sales = solution[gameweek]["sales"]
        budget = solution[gameweek]["budget"][0]
        free_transfers = solution[gameweek]["free_transfers"][0]
        paid_transfers = solution[gameweek]["paid_transfers"][0]

        print(f"Gameweek {gameweek} plan:")

        # Print purchases and sales
        if not purchases and not sales:
            print("- No transfers made")
        else:
            purchases = sorted(purchases, key=lambda p: element_types[p, gameweek])
            sales = sorted(sales, key=lambda p: element_types[p, gameweek])

            table = []
            for p1, p2 in zip(purchases, sales, strict=True):
                table.append({"Buy": web_names[p1], "Sell": web_names[p2]})

            print_table(table)

        # Print budget, free transfers, and paid transfers
        print(f"- Bank: {format_currency(budget)}")
        print(f"- Free transfers: {int(free_transfers)}")
        print(f"- Paid transfers: {int(paid_transfers)}")
        print()
