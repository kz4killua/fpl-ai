from collections import defaultdict

import polars as pl

from game.rules import DEF, ELEMENT_TYPES, FWD, GKP, MID, MNG
from game.utils import format_currency
from loaders.fpl import load_fixtures, load_static_elements, load_static_teams
from loaders.merged import load_merged
from loaders.utils import get_mapper, get_seasons, print_table

from .utils import (
    calculate_budget,
    calculate_points,
    calculate_team_value,
    calculate_transfer_cost,
    count_transfers,
    get_purchase_prices,
    get_selling_prices,
    load_results,
    make_automatic_substitutions,
    make_random_squad,
    update_free_transfers,
    update_purchase_prices,
)


class Simulator:
    def __init__(self, season: int):
        self.season = season

        # Load fixture data
        self.fixtures = load_fixtures([self.season]).collect()
        self.gameweeks = self.fixtures.get_column("gameweek").unique().sort().to_list()
        self.first_gameweek = self.gameweeks[0]
        self.last_gameweek = self.gameweeks[-1]

        # Initialize a team
        static_elements = load_static_elements(self.season, self.first_gameweek)
        static_elements = static_elements.collect()
        self.squad, self.budget = make_random_squad(static_elements)
        self.purchase_prices = get_purchase_prices(self.squad, static_elements)
        self.free_transfers = 0
        self.season_points = 0

        # Load results
        self.results = load_results(self.season).collect()

        # Update data for the first gameweek
        self.next_gameweek = self.first_gameweek
        self.reload()

    def reload(self):
        """Reload data for the next gameweek."""

        # Load static data for the next gameweek
        self.static_elements = load_static_elements(
            self.season, self.next_gameweek
        ).collect()
        self.static_teams = load_static_teams(self.season, self.next_gameweek).collect()
        self.static_players = self.static_elements.filter(
            pl.col("element_type").is_in([GKP, DEF, MID, FWD])
        )
        self.static_managers = self.static_elements.filter(
            pl.col("element_type") == MNG
        )

        # Recalculate player selling prices
        now_costs = get_mapper(self.static_elements, "id", "now_cost")
        self.selling_prices = get_selling_prices(
            self.squad, self.purchase_prices, now_costs
        )

        # Load historical and upcoming data
        seasons = get_seasons(self.season, 2)
        upcoming_gameweeks = list(range(self.next_gameweek, self.last_gameweek + 1))
        self.players, self.matches, self.managers = load_merged(
            seasons, self.season, upcoming_gameweeks
        )
        self.players = self.players.collect()
        self.matches = self.matches.collect()
        self.managers = self.managers.collect()

    def update(self, roles: dict, wildcard_gameweeks: list[int], log: bool = False):
        """Updates the squad and results for the next gameweek."""

        # Check that the season is not over
        if self.next_gameweek is None:
            raise ValueError("No gameweeks left to simulate.")

        # Get the results of the gameweek
        gameweek_results = self.results.filter(pl.col("gameweek") == self.next_gameweek)
        minutes = get_mapper(gameweek_results, "element", "minutes")
        total_points = get_mapper(gameweek_results, "element", "total_points")
        element_types = get_mapper(self.static_elements, "id", "element_type")
        now_costs = get_mapper(self.static_elements, "id", "now_cost")
        web_names = get_mapper(self.static_elements, "id", "web_name")

        # Calculate points scored by the squad
        substituted_roles = make_automatic_substitutions(roles, minutes, element_types)
        total_points = defaultdict(lambda: 0, total_points)
        gameweek_points = calculate_points(substituted_roles, total_points)

        # Update the budget, purchase prices, and free transfers
        new_squad = {
            *roles["starting_xi"],
            roles["reserve_gkp"],
            roles["reserve_out_1"],
            roles["reserve_out_2"],
            roles["reserve_out_3"],
        }
        transfers_made = count_transfers(self.squad, new_squad)
        transfer_cost = calculate_transfer_cost(
            self.free_transfers,
            transfers_made,
            self.next_gameweek,
            wildcard_gameweeks,
        )
        new_budget = calculate_budget(
            self.squad, new_squad, self.budget, self.selling_prices, now_costs
        )
        new_purchase_prices = update_purchase_prices(
            new_squad, self.purchase_prices, now_costs
        )
        new_selling_prices = get_selling_prices(
            new_squad, new_purchase_prices, now_costs
        )
        new_team_value = calculate_team_value(new_squad, new_selling_prices, new_budget)
        new_free_transfers = update_free_transfers(
            self.free_transfers,
            transfers_made,
            self.next_gameweek,
            wildcard_gameweeks,
        )

        # Update the overall points tally
        self.season_points += gameweek_points - transfer_cost

        if log:
            print_gameweek_summary(
                self.next_gameweek,
                gameweek_points,
                roles,
                substituted_roles,
                self.squad,
                new_squad,
                new_budget,
                new_team_value,
                transfer_cost,
                self.free_transfers,
                element_types,
                web_names,
                total_points,
                now_costs,
                self.selling_prices,
            )

        # Update the team state
        self.squad = new_squad
        self.budget = new_budget
        self.purchase_prices = new_purchase_prices
        self.free_transfers = new_free_transfers

        # Update data for the next gameweek
        current_index = self.gameweeks.index(self.next_gameweek)
        if current_index < len(self.gameweeks) - 1:
            self.next_gameweek = self.gameweeks[current_index + 1]
            self.reload()
        else:
            self.next_gameweek = None


def print_gameweek_summary(
    gameweek: int,
    gameweek_points: int,
    selected_roles: dict,
    substituted_roles: dict,
    initial_squad: set,
    final_squad: set,
    final_budget: int,
    final_team_value: int,
    transfer_cost: int,
    free_transfers: int,
    element_types: dict,
    web_names: dict,
    total_points: dict,
    now_costs: dict,
    selling_prices: dict,
):
    """Prints a report of the gameweek's activity and performance."""

    print(f"Gameweek {gameweek}: {gameweek_points} points")

    # Print the starting XI, including substitutions and captains
    print("Starting XI:")

    table = []
    for player in sorted(substituted_roles["starting_xi"], key=element_types.get):
        row = {
            "Position": ELEMENT_TYPES[element_types[player]],
            "Name": web_names[player],
            "Points": total_points[player],
        }

        if player not in selected_roles["starting_xi"]:
            row["Name"] = "-> " + row["Name"]
        if player == substituted_roles["captain"]:
            row["Name"] = "(C) " + row["Name"]
        if player == substituted_roles["vice_captain"]:
            row["Name"] = "(V) " + row["Name"]

        table.append(row)

    print_table(table)

    # Print the reserve players, including substitutions and captains
    print("Reserves:")

    table = []
    for player in [
        substituted_roles["reserve_gkp"],
        substituted_roles["reserve_out_1"],
        substituted_roles["reserve_out_2"],
        substituted_roles["reserve_out_3"],
    ]:
        row = {
            "Position": ELEMENT_TYPES[element_types[player]],
            "Name": web_names[player],
            "Points": total_points[player],
        }

        if player in selected_roles["starting_xi"]:
            row["Name"] = "<- " + row["Name"]
        if player == selected_roles["captain"]:
            row["Name"] = "(*C) " + row["Name"]
        if player == selected_roles["vice_captain"]:
            row["Name"] = "(*V) " + row["Name"]

        table.append(row)

    print_table(table)

    # Print transfer activity and final budget
    print(f"Transfers ({transfer_cost} points) [Free transfers: {free_transfers}]")

    buy = set(final_squad) - set(initial_squad)
    sell = set(initial_squad) - set(final_squad)
    buy = sorted(buy, key=lambda x: element_types[x])
    sell = sorted(sell, key=lambda x: element_types[x])

    table = []
    for p1, p2 in zip(buy, sell, strict=True):
        row = {
            "Buy": f"{web_names[p1]} ({format_currency(now_costs[p1])})",
            "Sell": f"{web_names[p2]} ({format_currency(selling_prices[p2])})",
        }

        table.append(row)

    print_table(table)

    print(f"Bank: {format_currency(final_budget)}")
    print(f"Team value: {format_currency(final_team_value)}")
    print()
