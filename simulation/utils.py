from collections import defaultdict
from collections.abc import Iterable
from copy import deepcopy

import polars as pl

from datautil.load.fpl import load_elements
from datautil.utils import get_mapper
from game.rules import (
    DEF,
    FWD,
    GKP,
    MAX_FREE_TRANSFERS,
    MAX_PLAYERS_PER_TEAM,
    MAX_STARTING_XI_DEFS,
    MAX_STARTING_XI_FWDS,
    MAX_STARTING_XI_GKPS,
    MAX_STARTING_XI_MIDS,
    MID,
    MIN_STARTING_XI_DEFS,
    MIN_STARTING_XI_FWDS,
    MIN_STARTING_XI_GKPS,
    MIN_STARTING_XI_MIDS,
    NUM_SQUAD_DEFS,
    NUM_SQUAD_FWDS,
    NUM_SQUAD_GKPS,
    NUM_SQUAD_MIDS,
    STARTING_BUDGET,
    TRANSFER_COST,
)


def make_random_squad(static_elements: pl.DataFrame) -> tuple[list[int], int]:
    """Return a random (legal) FPL squad and its budget."""

    while True:
        # Randomly select 2 GKPs, 5 DEFs, 5 MIDs, and 3 FWDs
        gkps = (
            static_elements.filter(pl.col("element_type") == GKP)
            .sample(NUM_SQUAD_GKPS)
            .get_column("id")
            .to_list()
        )
        defs = (
            static_elements.filter(pl.col("element_type") == DEF)
            .sample(NUM_SQUAD_DEFS)
            .get_column("id")
            .to_list()
        )
        mids = (
            static_elements.filter(pl.col("element_type") == MID)
            .sample(NUM_SQUAD_MIDS)
            .get_column("id")
            .to_list()
        )
        fwds = (
            static_elements.filter(pl.col("element_type") == FWD)
            .sample(NUM_SQUAD_FWDS)
            .get_column("id")
            .to_list()
        )
        squad = gkps + defs + mids + fwds

        # Filter the dataframe
        filtered_elements = static_elements.filter(pl.col("id").is_in(squad))
        assert len(filtered_elements) == len(squad)

        # Ensure that there are no more than 3 players from the same team
        team_counts = filtered_elements.get_column("team").value_counts()
        if team_counts.get_column("count").max() > MAX_PLAYERS_PER_TEAM:
            continue

        # Ensure that the total cost of the squad is within budget
        budget = STARTING_BUDGET - filtered_elements.get_column("now_cost").sum()
        if budget < 0:
            continue

        return squad, budget


def make_automatic_substitutions(roles: dict, minutes: dict, positions: dict) -> dict:
    """Return squad roles after making automatic substitutions."""

    roles = deepcopy(roles)
    minutes = defaultdict(lambda: 0, minutes)

    # Store the minimum and maximum number of players per position
    starting_xi_position_minimums = {
        GKP: MIN_STARTING_XI_GKPS,
        DEF: MIN_STARTING_XI_DEFS,
        MID: MIN_STARTING_XI_MIDS,
        FWD: MIN_STARTING_XI_FWDS,
    }
    starting_xi_position_maximums = {
        GKP: MAX_STARTING_XI_GKPS,
        DEF: MAX_STARTING_XI_DEFS,
        MID: MAX_STARTING_XI_MIDS,
        FWD: MAX_STARTING_XI_FWDS,
    }

    # Count the number of starting players in each position
    starting_xi_position_counts = {GKP: 0, DEF: 0, MID: 0, FWD: 0}
    for player in roles["starting_xi"]:
        starting_xi_position_counts[positions[player]] += 1

    for i, player in enumerate(roles["starting_xi"]):
        # Skip all players who played in the gameweek.
        if minutes[player] > 0:
            continue

        # Substitute with the next valid player.
        for key in ["reserve_gkp", "reserve_out_1", "reserve_out_2", "reserve_out_3"]:
            candidate = roles[key]

            # Skip reserves who played no minutes.
            if minutes[candidate] == 0:
                continue

            # If working with different positions, ensure the swap is legal.
            if positions[candidate] != positions[player] and (
                (
                    starting_xi_position_counts[positions[candidate]]
                    >= starting_xi_position_maximums[positions[candidate]]
                )
                or (
                    starting_xi_position_counts[positions[player]]
                    <= starting_xi_position_minimums[positions[player]]
                )
            ):
                continue

            # Swap the players.
            roles["starting_xi"][i], roles[key] = (roles[key], roles["starting_xi"][i])

            # Update position counts.
            starting_xi_position_counts[positions[candidate]] += 1
            starting_xi_position_counts[positions[player]] -= 1

            break

    # Replace the captain if necessary
    if (minutes[roles["captain"]] == 0) and (minutes[roles["vice_captain"]] > 0):
        roles["captain"], roles["vice_captain"] = (
            roles["vice_captain"],
            roles["captain"],
        )

    return roles


def load_results(season: str) -> pl.LazyFrame:
    """Load player points and minutes for each gameweek in the given season."""
    elements = load_elements([season])
    results = (
        elements.group_by(["season", "element", "round"])
        .agg(pl.col("total_points").sum(), pl.col("minutes").sum())
        .select(
            pl.col("season"),
            pl.col("element"),
            pl.col("round"),
            pl.col("total_points"),
            pl.col("minutes"),
        )
    )
    return results


def calculate_points(roles: dict, total_points: dict) -> int:
    """Calculate the total points scored by the squad in a gameweek."""
    points = 0
    for player in roles["starting_xi"]:
        if player == roles["captain"]:
            points += total_points[player] * 2
        else:
            points += total_points[player]
    return points


def calculate_budget(
    old_squad: Iterable[int],
    new_squad: Iterable[int],
    old_budget: int,
    selling_prices: dict,
    now_costs: dict,
) -> int:
    """Calculate the new budget after moving from an old to a new squad."""
    transfers_in = set(new_squad) - set(old_squad)
    transfers_out = set(old_squad) - set(new_squad)

    value = old_budget
    for player_out in transfers_out:
        value += selling_prices[player_out]
    for player_in in transfers_in:
        value -= now_costs[player_in]

    return value


def count_transfers(old_squad: Iterable[int], new_squad: Iterable[int]) -> int:
    """Count the number of transfers made to move between two squads."""
    return len(set(old_squad) - set(new_squad))


def calculate_transfer_cost(
    free_transfers: int,
    transfers_made: int,
    next_gameweek: int,
    wildcard_gameweeks: list,
) -> int:
    """Calculate the transfer cost for a gameweek."""
    # Transfers are free when wildcards are active and on the first gameweek
    if (next_gameweek == 1) or (next_gameweek in wildcard_gameweeks):
        return 0
    return max(transfers_made - free_transfers, 0) * TRANSFER_COST


def update_free_transfers(
    free_transfers: int,
    transfers_made: int,
    gameweek: int,
    wildcard_gameweeks: list,
):
    """Calculate the updated number of free transfers for a gameweek."""
    if gameweek in wildcard_gameweeks:
        return free_transfers
    # In the general case, FT2 = min(max(FT1 - TM1 + 1), 5)
    free_transfers = max(free_transfers - transfers_made + 1, 1)
    free_transfers = min(free_transfers, MAX_FREE_TRANSFERS)
    return free_transfers


def update_purchase_prices(
    squad: Iterable[int], purchase_prices: dict, now_costs: dict
) -> dict:
    """Update the purchase prices of players in the squad."""
    purchase_prices = purchase_prices.copy()
    # Remove players not in the squad
    for player in list(purchase_prices.keys()):
        if player not in squad:
            del purchase_prices[player]
    # Add new players to the purchase prices
    for player in squad:
        if player not in purchase_prices:
            purchase_prices[player] = now_costs[player]
    return purchase_prices


def get_purchase_prices(squad: list[int], static_elements: pl.DataFrame) -> dict:
    """Returns a dictionary mapping each player to their purchase price."""
    filtered_elements = static_elements.filter(pl.col("id").is_in(squad))
    purchase_prices = get_mapper(filtered_elements, "id", "now_cost")
    return purchase_prices


def get_selling_prices(
    squad: list[int], purchase_prices: dict, now_costs: dict
) -> dict:
    """Returns a dictionary mapping each player to their selling price."""
    selling_prices = dict()
    for player in squad:
        selling_prices[player] = calculate_selling_price(
            purchase_prices[player], now_costs[player]
        )
    return selling_prices


def calculate_selling_price(purchase_price: int, now_cost: int) -> int:
    """Calculates the selling price for a single player."""
    if now_cost <= purchase_price:
        return now_cost
    # Apply a 50% sell-on fee to any profits
    profit = (now_cost - purchase_price) // 2
    return purchase_price + profit


def calculate_team_value(squad: Iterable[int], selling_prices: dict, budget: int):
    """Returns the overall worth of the team (including budget)."""
    return budget + sum(selling_prices[player] for player in squad)
