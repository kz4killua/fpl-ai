from game.rules import DEF, FWD, GKP, MID
from game.utils import format_currency
from loaders.utils import print_table

from .solver import calculate_decayed_sum, solve


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
    parameters: dict[str, float],
    log: bool = False,
):
    """Runs optimization and returns the optimal squad roles for the next gameweek."""

    # Prepare optimization data
    initial_squad = set(initial_squad)
    players = list(
        set(p for (p, _) in total_points if element_types[p] in {GKP, DEF, MID, FWD})
    )
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

    # Filter players to speed up optimization
    players = filter_players(
        initial_squad=initial_squad,
        players=players,
        gameweeks=upcoming_gameweeks,
        element_types=element_types,
        total_points=total_points,
        round_decay=parameters["round_decay"],
        percentile=parameters["filter_percentile"],
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
        round_decay=parameters["round_decay"],
        starting_xi_multiplier=parameters["starting_xi_multiplier"],
        captain_multiplier=parameters["captain_multiplier"],
        vice_captain_multiplier=parameters["vice_captain_multiplier"],
        reserve_gkp_multiplier=parameters["reserve_gkp_multiplier"],
        reserve_out_1_multiplier=parameters["reserve_out_1_multiplier"],
        reserve_out_2_multiplier=parameters["reserve_out_2_multiplier"],
        reserve_out_3_multiplier=parameters["reserve_out_3_multiplier"],
        budget_value=parameters["budget_value"],
        free_transfer_value=parameters["free_transfer_value"],
        transfer_cost_multiplier=parameters["transfer_cost_multiplier"],
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


def filter_players(
    initial_squad: set[int],
    players: list[int],
    gameweeks: list[int],
    element_types: dict[tuple[int, int], int],
    total_points: dict[tuple[int, int], float],
    round_decay: float,
    percentile: float,
):
    """Reduce the player pool to speed up optimization."""

    # Use a weighted sum of predicted points to rank players
    sort_keys = {}
    for player in players:
        player_points = []
        for gameweek in gameweeks:
            player_points.append(total_points.get((player, gameweek), 0))
        sort_keys[player] = calculate_decayed_sum(player_points, round_decay)

    # Keep all players in the initial squad
    filtered_players = set(initial_squad)

    # Perform filtering across element types to ensure coverage
    groups = {GKP: [], DEF: [], MID: [], FWD: []}
    for p in players:
        groups[element_types[p, gameweeks[0]]].append(p)

    # Filter the top players in each group
    for element_type in groups:
        k = int(len(groups[element_type]) * percentile)
        sorted_players = sorted(
            groups[element_type], key=lambda p: sort_keys[p], reverse=True
        )
        filtered_players.update(sorted_players[:k])

    return list(filtered_players)
