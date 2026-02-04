import uuid

from ortools.linear_solver import pywraplp

from game.rules import (
    DEF,
    FWD,
    GKP,
    MAX_FREE_TRANSFERS,
    MAX_PLAYERS_PER_TEAM,
    MID,
    MIN_STARTING_XI_DEFS,
    MIN_STARTING_XI_FWDS,
    NUM_SQUAD_DEFS,
    NUM_SQUAD_FWDS,
    NUM_SQUAD_GKPS,
    NUM_SQUAD_MIDS,
    NUM_SQUAD_PLAYERS,
    NUM_STARTING_XI,
    NUM_STARTING_XI_GKPS,
    TRANSFER_COST,
)


def solve(
    # Initial conditions
    initial_squad: set[int],
    initial_budget: int,
    initial_free_transfers: int,
    # Players & gameweeks
    players: list[int],
    gameweeks: list[int],
    # Chip usage
    wildcards: list[int],
    # Player attributes
    total_points: dict[tuple[int, int], float],
    teams: dict[tuple[int, int], int],
    element_types: dict[tuple[int, int], int],
    now_costs: dict[tuple[int, int], int],
    selling_prices: dict[tuple[int, int], int],
    # Optimization parameters
    round_decay: float,
    starting_xi_multiplier: float,
    captain_multiplier: float,
    vice_captain_multiplier: float,
    reserve_gkp_multiplier: float,
    reserve_out_1_multiplier: float,
    reserve_out_2_multiplier: float,
    reserve_out_3_multiplier: float,
    budget_value: float,
    free_transfer_value: dict[int, float],
    transfer_cost_multiplier: float,
    # Logging
    log: bool = False,
):
    """Solve the optimization problem to find the best squad configuration."""

    solver = pywraplp.Solver.CreateSolver("CBC_MIXED_INTEGER_PROGRAMMING")
    if not solver:
        raise RuntimeError("Solver not found.")

    # Check for problematic inputs
    check_inputs(
        initial_squad,
        initial_budget,
        initial_free_transfers,
        players,
        gameweeks,
        wildcards,
        total_points,
        teams,
        element_types,
        now_costs,
        selling_prices,
    )

    # Set up the optimization problem
    variables = create_variables(solver, players, gameweeks)
    create_constraints(
        solver,
        variables,
        initial_squad,
        initial_budget,
        initial_free_transfers,
        players,
        gameweeks,
        teams,
        element_types,
        now_costs,
        selling_prices,
        wildcards,
    )
    create_objective(
        solver,
        variables,
        players,
        gameweeks,
        total_points,
        round_decay,
        starting_xi_multiplier,
        captain_multiplier,
        vice_captain_multiplier,
        reserve_gkp_multiplier,
        reserve_out_1_multiplier,
        reserve_out_2_multiplier,
        reserve_out_3_multiplier,
        budget_value,
        free_transfer_value,
        transfer_cost_multiplier,
    )

    # Solve the optimization problem
    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        raise RuntimeError(f"Solver failed with status {status}.")

    if log:
        print(f"Solver time: {solver.wall_time():d}ms")
        print(f"Objective value: {solver.Objective().Value()}")
        print(f"Optimal: {status == pywraplp.Solver.OPTIMAL}")
        print()

    return get_solution(variables, gameweeks)


def create_variables(
    solver: pywraplp.Solver,
    players: list[int],
    gameweeks: list[int],
) -> dict:
    """Create the decision variables for the optimization problem."""
    variables = dict()

    # Add variables for squad selection
    variables["squad"] = {
        (p, g): solver.IntVar(0, 1, f"squad_{p}_{g}")
        for p in players
        for g in gameweeks
    }
    variables["captain"] = {
        (p, g): solver.IntVar(0, 1, f"captain_{p}_{g}")
        for p in players
        for g in gameweeks
    }
    variables["vice_captain"] = {
        (p, g): solver.IntVar(0, 1, f"vice_captain_{p}_{g}")
        for p in players
        for g in gameweeks
    }
    variables["starting_xi"] = {
        (p, g): solver.IntVar(0, 1, f"starting_xi_{p}_{g}")
        for p in players
        for g in gameweeks
    }
    variables["reserve_gkp"] = {
        (p, g): solver.IntVar(0, 1, f"reserve_gkp_{p}_{g}")
        for p in players
        for g in gameweeks
    }
    variables["reserve_out_1"] = {
        (p, g): solver.IntVar(0, 1, f"reserve_out_1_{p}_{g}")
        for p in players
        for g in gameweeks
    }
    variables["reserve_out_2"] = {
        (p, g): solver.IntVar(0, 1, f"reserve_out_2_{p}_{g}")
        for p in players
        for g in gameweeks
    }
    variables["reserve_out_3"] = {
        (p, g): solver.IntVar(0, 1, f"reserve_out_3_{p}_{g}")
        for p in players
        for g in gameweeks
    }

    # Add variables for transfers
    variables["paid_transfers"] = {
        g: solver.IntVar(0, 15, f"paid_transfers_{g}") for g in gameweeks
    }
    variables["free_transfers"] = {
        g: solver.IntVar(0, MAX_FREE_TRANSFERS, f"free_transfers_{g}")
        for g in gameweeks
    }
    for i in range(MAX_FREE_TRANSFERS + 1):
        variables[f"free_transfers_{i}"] = {
            g: solver.IntVar(0, 1, f"free_transfers_{i}_{g}") for g in gameweeks
        }

    # Add variables for budget, purchases, and sales
    variables["budget"] = {
        g: solver.IntVar(0, 999_999, f"budget_{g}") for g in gameweeks
    }
    variables["purchases"] = {
        (p, g): solver.IntVar(0, 1, f"purchase_{p}_{g}")
        for p in players
        for g in gameweeks
    }
    variables["sales"] = {
        (p, g): solver.IntVar(0, 1, f"sale_{p}_{g}") for p in players for g in gameweeks
    }

    # Add variables for chips
    variables["wildcards"] = {
        g: solver.IntVar(0, 1, f"wildcard_{g}") for g in gameweeks
    }

    return variables


def create_constraints(
    solver: pywraplp.Solver,
    variables: dict,
    # Initial conditions
    initial_squad: set[int],
    initial_budget: int,
    initial_free_transfers: int,
    # Players & gameweeks
    players: list[int],
    gameweeks: list[int],
    # Player attributes
    teams: dict[tuple[int, int], int],
    element_types: dict[tuple[int, int], int],
    now_costs: dict[tuple[int, int], int],
    selling_prices: dict[tuple[int, int], int],
    # Chip usage
    wildcards: list[int],
):
    """Create the constraints for the optimization problem."""

    # Squad selection
    # ---------------

    # There must be exactly 2 GKPs, 5 DEFs, 5 MIDs, and 3 FWDS in the squad (15 players)
    for g in gameweeks:
        solver.Add(
            sum(variables["squad"][p, g] for p in players if element_types[p, g] == GKP)
            == NUM_SQUAD_GKPS
        )
        solver.Add(
            sum(variables["squad"][p, g] for p in players if element_types[p, g] == DEF)
            == NUM_SQUAD_DEFS
        )
        solver.Add(
            sum(variables["squad"][p, g] for p in players if element_types[p, g] == MID)
            == NUM_SQUAD_MIDS
        )
        solver.Add(
            sum(variables["squad"][p, g] for p in players if element_types[p, g] == FWD)
            == NUM_SQUAD_FWDS
        )

    # There must be exactly 11 starting XI players
    for g in gameweeks:
        solver.Add(
            sum(variables["starting_xi"][p, g] for p in players) == NUM_STARTING_XI
        )

    # There must be exactly 1 starting GKP, at least 3 starting DEFs,
    # and at least 1 starting FWD
    for g in gameweeks:
        solver.Add(
            sum(
                variables["starting_xi"][p, g]
                for p in players
                if element_types[p, g] == GKP
            )
            == NUM_STARTING_XI_GKPS
        )
        solver.Add(
            sum(
                variables["starting_xi"][p, g]
                for p in players
                if element_types[p, g] == DEF
            )
            >= MIN_STARTING_XI_DEFS
        )
        solver.Add(
            sum(
                variables["starting_xi"][p, g]
                for p in players
                if element_types[p, g] == FWD
            )
            >= MIN_STARTING_XI_FWDS
        )

    # There must be exactly 1 reserve GKP
    for g in gameweeks:
        solver.Add(
            sum(
                variables["reserve_gkp"][p, g]
                for p in players
                if element_types[p, g] == GKP
            )
            == 1
        )

    # There must be exactly 3 reserve outfield players
    for g in gameweeks:
        solver.Add(
            sum(
                variables["reserve_out_1"][p, g]
                for p in players
                if element_types[p, g] != GKP
            )
            == 1
        )
        solver.Add(
            sum(
                variables["reserve_out_2"][p, g]
                for p in players
                if element_types[p, g] != GKP
            )
            == 1
        )
        solver.Add(
            sum(
                variables["reserve_out_3"][p, g]
                for p in players
                if element_types[p, g] != GKP
            )
            == 1
        )

    # There must be exactly 1 captain and 1 vice-captain
    for g in gameweeks:
        solver.Add(sum(variables["captain"][p, g] for p in players) == 1)
        solver.Add(sum(variables["vice_captain"][p, g] for p in players) == 1)

    # The captain and vice-captain must be in the starting XI
    for g in gameweeks:
        for p in players:
            solver.Add(variables["captain"][p, g] <= variables["starting_xi"][p, g])
            solver.Add(
                variables["vice_captain"][p, g] <= variables["starting_xi"][p, g]
            )

    # The starting XI must be in the squad
    for g in gameweeks:
        for p in players:
            solver.Add(variables["starting_xi"][p, g] <= variables["squad"][p, g])

    # The reserve GKP must be in the squad
    for g in gameweeks:
        for p in players:
            solver.Add(variables["reserve_gkp"][p, g] <= variables["squad"][p, g])

    # The reserve outfield players must be in the squad
    for g in gameweeks:
        for p in players:
            solver.Add(variables["reserve_out_1"][p, g] <= variables["squad"][p, g])
            solver.Add(variables["reserve_out_2"][p, g] <= variables["squad"][p, g])
            solver.Add(variables["reserve_out_3"][p, g] <= variables["squad"][p, g])

    # The captain and vice-captain must be distinct
    for g in gameweeks:
        for p in players:
            solver.Add(
                variables["captain"][p, g] + variables["vice_captain"][p, g] <= 1
            )

    # The reserve GKP and reserve outfield players must distinct
    for g in gameweeks:
        for p in players:
            solver.Add(
                variables["reserve_gkp"][p, g]
                + variables["reserve_out_1"][p, g]
                + variables["reserve_out_2"][p, g]
                + variables["reserve_out_3"][p, g]
                <= 1
            )

    # The starting XI, reserve GKP, and reserve players must be distinct
    for g in gameweeks:
        for p in players:
            solver.Add(
                variables["starting_xi"][p, g]
                + variables["reserve_gkp"][p, g]
                + variables["reserve_out_1"][p, g]
                + variables["reserve_out_2"][p, g]
                + variables["reserve_out_3"][p, g]
                <= 1
            )

    # There must be no more than 3 players from the same team in the squad
    for g in gameweeks:
        for t in set(teams.values()):
            solver.Add(
                sum(variables["squad"][p, g] for p in players if teams[p, g] == t)
                <= MAX_PLAYERS_PER_TEAM
            )

    # Transfers, budget, and wildcards
    # --------------------------------

    # The budget must be consistent with purchases and sales
    for i, g in enumerate(gameweeks):
        prior_budget = (
            initial_budget if i == 0 else variables["budget"][gameweeks[i - 1]]
        )
        income = sum(variables["sales"][p, g] * selling_prices[p, g] for p in players)
        expenses = sum(variables["purchases"][p, g] * now_costs[p, g] for p in players)
        solver.Add(variables["budget"][g] == prior_budget + income - expenses)

    # The squad must be consistent with purchases and sales
    for i, g in enumerate(gameweeks):
        for p in players:
            # Set the initial ownership for the first gameweek
            prior_owned = (
                int(p in initial_squad)
                if i == 0
                else variables["squad"][p, gameweeks[i - 1]]
            )

            # Purchases add to the squad, sales remove from the squad
            solver.Add(
                variables["squad"][p, g]
                == (
                    prior_owned
                    + variables["purchases"][p, g]
                    - variables["sales"][p, g]
                )
            )

            # Sell only owned players, buy only unowned players
            solver.Add(variables["sales"][p, g] <= prior_owned)
            solver.Add(variables["purchases"][p, g] <= 1 - prior_owned)

    # Activate wildcards in selected gameweeks
    for g in gameweeks:
        solver.Add(variables["wildcards"][g] == int(g in wildcards))

    # Update the number of paid transfers
    for g in gameweeks:
        # In gameweek 1 and on wildcard gameweeks, no transfers are paid
        if g == 1 or g in wildcards:
            solver.Add(variables["paid_transfers"][g] == 0)

        # Otherwise: paid_transfers = max(transfers_made - free_transfers, 0)
        else:
            paid_transfers = variables["paid_transfers"][g]
            free_transfers = variables["free_transfers"][g]
            transfers_made = sum(variables["purchases"][p, g] for p in players)
            create_max_constraint(
                solver, paid_transfers, transfers_made - free_transfers, 0, 15
            )

    # Update the number of free transfers
    for i, g in enumerate(gameweeks):
        # Set the initial number of free transfers
        if i == 0:
            solver.Add(variables["free_transfers"][g] == initial_free_transfers)

        # Free transfers are rolled over after wildcards
        elif gameweeks[i - 1] in wildcards:
            solver.Add(
                variables["free_transfers"][g]
                == variables["free_transfers"][gameweeks[i - 1]]
            )

        # Otherwise:
        # FT2 = min(max(1, FT1 - TM1 + 1), 5)
        # where:
        # - TM1 is the number of transfers made in the previous gameweek
        # - FT1 is the number of free transfers in the previous gameweek
        # - FT2 is the number of free transfers in the current gameweek
        # Inspired by @sertalpbilal: https://youtu.be/Prv8M7hE3vk?t=2079
        else:
            ft2 = variables["free_transfers"][g]
            ft1 = variables["free_transfers"][gameweeks[i - 1]]
            tm1 = sum(variables["purchases"][p, gameweeks[i - 1]] for p in players)

            # Enforce the inner max constraint
            inner = solver.IntVar(1, 15, f"_{uuid.uuid4()}")
            create_max_constraint(solver, inner, 1, ft1 - tm1 + 1, 15)

            # Enforce the outer min constraint
            create_min_constraint(solver, ft2, inner, MAX_FREE_TRANSFERS, 15)

    for g in gameweeks:
        # Ensure exactly one free transfer count is selected
        solver.Add(
            sum(
                variables[f"free_transfers_{i}"][g]
                for i in range(MAX_FREE_TRANSFERS + 1)
            )
            == 1
        )
        # Link the free transfer count to the corresponding variable
        solver.Add(
            variables["free_transfers"][g]
            == sum(
                i * variables[f"free_transfers_{i}"][g]
                for i in range(MAX_FREE_TRANSFERS + 1)
            )
        )


def create_objective(
    solver: pywraplp.Solver,
    variables: dict,
    players: list[int],
    gameweeks: list[int],
    total_points: dict[tuple[int, int], float],
    # Optimization parameters
    round_decay: float,
    starting_xi_multiplier: float,
    captain_multiplier: float,
    vice_captain_multiplier: float,
    reserve_gkp_multiplier: float,
    reserve_out_1_multiplier: float,
    reserve_out_2_multiplier: float,
    reserve_out_3_multiplier: float,
    budget_value: float,
    free_transfer_value: dict[int, float],
    transfer_cost_multiplier: float,
):
    """Create the objective function for the optimization problem."""

    scores = []

    # Calculate the total scores for each gameweek
    for g in gameweeks:
        score = 0

        # Sum up the total points for the starting XI
        score += sum(
            variables["starting_xi"][p, g] * total_points[p, g] * starting_xi_multiplier
            for p in players
        )

        # Account for the captain and vice captain multipliers
        score += sum(
            variables["captain"][p, g]
            * total_points[p, g]
            * (captain_multiplier - starting_xi_multiplier)
            for p in players
        )
        score += sum(
            variables["vice_captain"][p, g]
            * total_points[p, g]
            * (vice_captain_multiplier - starting_xi_multiplier)
            for p in players
        )

        # Sum up the total points for all reserve players
        score += sum(
            variables["reserve_gkp"][p, g] * total_points[p, g] * reserve_gkp_multiplier
            for p in players
        )
        score += sum(
            variables["reserve_out_1"][p, g]
            * total_points[p, g]
            * reserve_out_1_multiplier
            for p in players
        )
        score += sum(
            variables["reserve_out_2"][p, g]
            * total_points[p, g]
            * reserve_out_2_multiplier
            for p in players
        )
        score += sum(
            variables["reserve_out_3"][p, g]
            * total_points[p, g]
            * reserve_out_3_multiplier
            for p in players
        )

        # Add transfer costs
        score -= (
            variables["paid_transfers"][g] * TRANSFER_COST * transfer_cost_multiplier
        )

        # Add the budget value
        score += variables["budget"][g] * budget_value

        # Add the free transfer value
        for i in range(MAX_FREE_TRANSFERS + 1):
            score += variables[f"free_transfers_{i}"][g] * free_transfer_value[i]

        scores.append(score)

    # Calculated the decay-weighed sum of scores
    objective = calculate_decayed_sum(scores, round_decay)

    # Set the objective to maximize the total score
    solver.Maximize(objective)


def get_solution(
    variables: dict[str, dict[tuple, pywraplp.Variable]], gameweeks: list[int]
) -> dict:
    """Extract the solution from the optimization variables."""

    solutions = dict()

    for gameweek in gameweeks:
        squad = [
            p
            for (p, g), v in variables["squad"].items()
            if g == gameweek and v.solution_value() == 1
        ]
        starting_xi = [
            p
            for (p, g), v in variables["starting_xi"].items()
            if g == gameweek and v.solution_value() == 1
        ]
        reserve_gkp = [
            p
            for (p, g), v in variables["reserve_gkp"].items()
            if g == gameweek and v.solution_value() == 1
        ]
        reserve_out_1 = [
            p
            for (p, g), v in variables["reserve_out_1"].items()
            if g == gameweek and v.solution_value() == 1
        ]
        reserve_out_2 = [
            p
            for (p, g), v in variables["reserve_out_2"].items()
            if g == gameweek and v.solution_value() == 1
        ]
        reserve_out_3 = [
            p
            for (p, g), v in variables["reserve_out_3"].items()
            if g == gameweek and v.solution_value() == 1
        ]
        captain = [
            p
            for (p, g), v in variables["captain"].items()
            if g == gameweek and v.solution_value() == 1
        ]
        vice_captain = [
            p
            for (p, g), v in variables["vice_captain"].items()
            if g == gameweek and v.solution_value() == 1
        ]
        purchases = [
            p
            for (p, g), v in variables["purchases"].items()
            if g == gameweek and v.solution_value() == 1
        ]
        sales = [
            p
            for (p, g), v in variables["sales"].items()
            if g == gameweek and v.solution_value() == 1
        ]
        budget = [
            v.solution_value() for g, v in variables["budget"].items() if g == gameweek
        ]
        free_transfers = [
            v.solution_value()
            for g, v in variables["free_transfers"].items()
            if g == gameweek
        ]
        paid_transfers = [
            v.solution_value()
            for g, v in variables["paid_transfers"].items()
            if g == gameweek
        ]

        solutions[gameweek] = {
            "squad": squad,
            "starting_xi": starting_xi,
            "reserve_gkp": reserve_gkp,
            "reserve_out_1": reserve_out_1,
            "reserve_out_2": reserve_out_2,
            "reserve_out_3": reserve_out_3,
            "captain": captain,
            "vice_captain": vice_captain,
            "purchases": purchases,
            "sales": sales,
            "budget": budget,
            "free_transfers": free_transfers,
            "paid_transfers": paid_transfers,
        }

    return solutions


def check_inputs(
    # Initial conditions
    initial_squad: set[int],
    initial_budget: int,
    initial_free_transfers: int,
    # Players & gameweeks
    players: list[int],
    gameweeks: list[int],
    # Chip usage
    wildcards: list[int],
    # Player attributes
    total_points: dict[tuple[int, int], float],
    teams: dict[tuple[int, int], int],
    element_types: dict[tuple[int, int], int],
    now_costs: dict[tuple[int, int], int],
    selling_prices: dict[tuple[int, int], int],
):
    """Run sanity checks on the inputs to catch errors."""
    if gameweeks != sorted(set(gameweeks)):
        raise ValueError("Gameweeks should be in sorted order.")
    if not all(p in players for p in initial_squad):
        raise ValueError("All squad players must be in the player list.")
    if len(set(initial_squad)) != NUM_SQUAD_PLAYERS:
        raise ValueError(f"Squad must contain exactly {NUM_SQUAD_PLAYERS} players.")
    if initial_budget < 0:
        raise ValueError("Initial budget cannot be negative.")
    if initial_free_transfers < 0 or initial_free_transfers > MAX_FREE_TRANSFERS:
        raise ValueError(
            f"Initial free transfers must be between 0 and {MAX_FREE_TRANSFERS}."
        )
    if 1 in wildcards:
        raise ValueError("Wildcards cannot be activated in gameweek 1.")
    if gameweeks[0] == 1 and initial_free_transfers != 0:
        raise ValueError("Free transfers are not available in gameweek 1.")
    if not all((p, g) in total_points for p in players for g in gameweeks):
        raise ValueError("Total points must be provided for all players and gameweeks.")
    if not all((p, g) in teams for p in players for g in gameweeks):
        raise ValueError("Teams must be provided for all players and gameweeks.")
    if not all((p, g) in element_types for p in players for g in gameweeks):
        raise ValueError(
            "Element types must be provided for all players and gameweeks."
        )
    if not all((p, g) in now_costs for p in players for g in gameweeks):
        raise ValueError("Now costs must be provided for all players and gameweeks.")
    if not all((p, g) in selling_prices for p in players for g in gameweeks):
        raise ValueError(
            "Selling prices must be provided for all players and gameweeks."
        )


def calculate_decayed_sum(
    scores: list[float] | list[pywraplp.Variable],
    decay: float,
):
    """Sum the scores with decay weights."""
    weights = [decay**i for i in range(len(scores))]
    total = sum(weights)
    weights = [w / total for w in weights]
    return sum(s * w for s, w in zip(scores, weights, strict=True))


def create_max_constraint(
    solver: pywraplp.Solver,
    y: pywraplp.Variable | pywraplp.LinearExpr,
    a: pywraplp.Variable | pywraplp.LinearExpr,
    b: pywraplp.Variable | pywraplp.LinearExpr,
    m: int,
    z: pywraplp.Variable | pywraplp.LinearExpr | None = None,
):
    """
    Enforce the constraint y = max(a, b).

    This constraint is equivalent to the following linear inequalities:
    - y >= a
    - y >= b
    - y <= a + m * (1 - z)
    - y <= b + m * z

    where:
    - m is a sufficiently large constant such that a, b <= m for any "reasonable"
        solution
    - z is a binary variable such that z = 1 if a >= b, and z = 0 otherwise

    If z is not provided, it is initialized as a binary variable.

    See: https://or.stackexchange.com/questions/711/how-to-formulate-linearize-a-maximum-function-in-a-constraint
    """

    # Create the binary variable z if not provided
    if z is None:
        z = solver.IntVar(0, 1, f"_{uuid.uuid4()}")

    # Add the linear inequalities
    solver.Add(y >= a)
    solver.Add(y >= b)
    solver.Add(y <= a + m * (1 - z))
    solver.Add(y <= b + m * z)


def create_min_constraint(
    solver: pywraplp.Solver,
    y: pywraplp.Variable | pywraplp.LinearExpr,
    a: pywraplp.Variable | pywraplp.LinearExpr,
    b: pywraplp.Variable | pywraplp.LinearExpr,
    m: int,
    z: pywraplp.Variable | pywraplp.LinearExpr | None = None,
):
    """
    Enforce the constraint y = min(a, b).

    This constraint is equivalent to the following linear inequalities:
    - y <= a
    - y <= b
    - y >= a - m * (1 - z)
    - y >= b - m * z

    where:
    - m is a sufficiently large constant such that a, b <= m for any "reasonable"
        solution
    - z is a binary variable such that z = 1 if a <= b, and z = 0 otherwise

    If z is not provided, it is initialized as a binary variable.

    See: https://or.stackexchange.com/questions/1160/how-to-linearize-min-function-as-a-constraint
    """

    # Create the binary variable z if not provided
    if z is None:
        z = solver.IntVar(0, 1, f"_{uuid.uuid4()}")

    # Add the linear inequalities
    solver.Add(y <= a)
    solver.Add(y <= b)
    solver.Add(y >= a - m * (1 - z))
    solver.Add(y >= b - m * z)
