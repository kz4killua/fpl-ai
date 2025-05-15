from collections import defaultdict

import polars as pl

from datautil.load.fpl import load_fixtures
from datautil.load.fplcache import load_static_elements, load_static_teams
from datautil.load.merged import load_merged
from datautil.upcoming import (
    get_upcoming_fixtures,
    get_upcoming_gameweeks,
    get_upcoming_manager_data,
    get_upcoming_player_data,
    get_upcoming_team_data,
)
from datautil.utils import get_seasons
from features.engineer_features import engineer_features
from optimization.optimize import optimize_squad
from optimization.parameters import OPTIMIZATION_WINDOW_SIZE
from optimization.rules import DEF, ELEMENT_TYPES, FWD, GKP, MID, MNG
from prediction.model import load_model
from prediction.predict import make_predictions
from utils.frames import get_mapper

from .utils import (
    calculate_budget,
    calculate_points,
    calculate_transfer_cost,
    count_transfers,
    get_purchase_prices,
    get_selling_prices,
    load_results,
    make_automatic_substitutions,
    make_random_squad,
    remove_upcoming_data,
    update_free_transfers,
    update_purchase_prices,
)


class Simulator:
    def __init__(self, season: str):
        self.season = season
        self.first_gameweek = 1
        self.last_gameweek = 38
        self.next_gameweek = self.first_gameweek
        self.season_points = 0
        self.initialize_data()
        self.initialize_team()

    def initialize_data(self):
        """Load data, fixtures, and results for the season."""
        seasons = get_seasons(self.season, 5)
        self.unfiltered_players, self.unfiltered_teams, self.unfiltered_managers = (
            load_merged(seasons)
        )
        self.fixtures = load_fixtures([self.season])
        self.results = load_results(self.season).collect()

    def initialize_team(self):
        """Create a random squad for the season."""
        # Use data from the first gameweek to construct the initial squad
        static_elements = load_static_elements(
            self.season, self.first_gameweek
        ).collect()
        self.free_transfers = 0
        self.squad, self.budget = make_random_squad(static_elements)
        self.purchase_prices = get_purchase_prices(self.squad, static_elements)

    @property
    def static_elements(self):
        """Load static elements for the current gameweek."""
        return load_static_elements(self.season, self.next_gameweek)
    
    @property
    def static_teams(self):
        """Load static teams for the current gameweek."""
        return load_static_teams(self.season, self.next_gameweek)

    @property
    def static_players(self):
        """Load static players for the current gameweek."""
        return self.static_elements.filter(
            pl.col("element_type").is_in([GKP, DEF, MID, FWD])
        )

    @property
    def static_managers(self):
        """Load static managers for the current gameweek."""
        return self.static_elements.filter(pl.col("element_type") == MNG)

    @property
    def historical_players(self):
        """Load historical players data up to the current gameweek."""
        return remove_upcoming_data(
            self.unfiltered_players, self.season, self.next_gameweek
        )

    @property
    def historical_teams(self):
        """Load historical teams data up to the current gameweek."""
        return remove_upcoming_data(
            self.unfiltered_teams, self.season, self.next_gameweek
        )

    @property
    def historical_managers(self):
        """Load historical managers data up to the current gameweek."""
        return remove_upcoming_data(
            self.unfiltered_managers, self.season, self.next_gameweek
        )

    @property
    def selling_prices(self):
        """Calculate the selling prices for the current squad."""
        static_elements_df = self.static_elements.collect()
        now_costs = get_mapper(static_elements_df, "id", "now_cost")
        return get_selling_prices(self.squad, self.purchase_prices, now_costs)

    def update(self, roles: dict, wildcard_gameweeks: list[int], log: bool = False):
        """Updates the squad and results for the next gameweek."""

        # Get the results of the gameweek
        gameweek_results = self.results.filter(pl.col("round") == self.next_gameweek)
        minutes = get_mapper(gameweek_results, "element", "minutes")
        total_points = get_mapper(gameweek_results, "element", "total_points")
        static_elements_df = self.static_elements.collect()
        element_types = get_mapper(static_elements_df, "id", "element_type")
        now_costs = get_mapper(static_elements_df, "id", "now_cost")
        web_names = get_mapper(static_elements_df, "id", "web_name")

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
                transfer_cost,
                self.free_transfers,
                element_types,
                web_names,
                total_points,
                now_costs,
                self.selling_prices,
            )

        # Update the variables for the next gameweek
        self.squad = new_squad
        self.budget = new_budget
        self.purchase_prices = new_purchase_prices
        self.free_transfers = new_free_transfers

        # Move to the next gameweek
        self.next_gameweek += 1

        # Skip cancelled gameweeks.
        if self.season == "2022-23" and self.next_gameweek == 7:
            self.next_gameweek = 8


def simulate(season: str, wildcard_gameweeks: list[int], log: bool = False) -> int:
    model = load_model("simulation_model")
    simulator = Simulator(season)

    # Simulate each gameweek
    while simulator.next_gameweek <= simulator.last_gameweek:
        # Unpack attributes from the simulator
        squad = simulator.squad
        purchase_prices = simulator.purchase_prices
        budget = simulator.budget
        free_transfers = simulator.free_transfers
        next_gameweek = simulator.next_gameweek
        last_gameweek = simulator.last_gameweek

        # Load data from the simulator
        static_elements = simulator.static_elements
        static_players = simulator.static_players
        static_teams = simulator.static_teams
        static_managers = simulator.static_managers
        historical_players = simulator.historical_players
        historical_teams = simulator.historical_teams
        historical_managers = simulator.historical_managers
        fixtures = simulator.fixtures

        # Get upcoming data
        upcoming_gameweeks = get_upcoming_gameweeks(
            next_gameweek, OPTIMIZATION_WINDOW_SIZE, last_gameweek
        )
        upcoming_fixtures = get_upcoming_fixtures(fixtures, season, upcoming_gameweeks)
        upcoming_players = get_upcoming_player_data(
            upcoming_fixtures, static_players, static_teams
        )
        upcoming_managers = get_upcoming_manager_data(
            upcoming_fixtures, static_managers, static_teams
        )
        upcoming_teams = get_upcoming_team_data(upcoming_fixtures, static_teams)

        # Combine the data
        combined_players = pl.concat(
            [historical_players, upcoming_players], how="diagonal_relaxed"
        )
        _ = pl.concat([historical_teams, upcoming_teams], how="diagonal_relaxed")
        _ = pl.concat([historical_managers, upcoming_managers], how="diagonal_relaxed")

        # Engineer features. Keep only those needed for prediction
        features = engineer_features(combined_players)
        features = features.filter(
            (pl.col("season") == season) & (pl.col("round").is_in(upcoming_gameweeks))
        )

        # Collect all lazy frames
        static_elements = static_elements.collect()
        features = features.collect()

        # Predict total points
        predictions = make_predictions(features, model)
        predictions = get_mapper(predictions, ["element", "round"], "total_points")
        for player in static_elements["id"]:
            for gameweek in upcoming_gameweeks:
                if (player, gameweek) not in predictions:
                    predictions[(player, gameweek)] = 0

        # Optimize the squad
        now_costs = get_mapper(static_elements, "id", "now_cost")
        element_types = get_mapper(static_elements, "id", "element_type")
        teams = get_mapper(static_elements, "id", "team")
        selling_prices = get_selling_prices(squad, purchase_prices, now_costs)

        roles = optimize_squad(
            squad,
            budget,
            free_transfers,
            now_costs,
            selling_prices,
            upcoming_gameweeks,
            wildcard_gameweeks,
            predictions,
            element_types,
            teams,
        )

        # Update the simulator
        simulator.update(roles, wildcard_gameweeks, log=log)

    return simulator.season_points


def print_gameweek_summary(
    gameweek: int,
    gameweek_points: int,
    selected_roles: dict,
    substituted_roles: dict,
    initial_squad: set,
    final_squad: set,
    final_budget: int,
    transfer_cost: int,
    free_transfers: int,
    element_types: dict,
    web_names: dict,
    total_points: dict,
    now_costs: dict,
    selling_prices: dict,
):
    """Prints a report of the gameweek's activity and performance."""

    element_type_names = ELEMENT_TYPES

    print(f"Gameweek {gameweek}: {gameweek_points} points")

    # Print the starting XI, including substitutions and captains
    print("Starting XI:")

    headers = ["", "", "Position", "Name", "Points"]
    data = []
    for player in sorted(substituted_roles["starting_xi"], key=element_types.get):
        row = []

        if player not in selected_roles["starting_xi"]:
            row.append("->")
        else:
            row.append("  ")

        if player == substituted_roles["captain"]:
            row.append("(C)")
        elif player == substituted_roles["vice_captain"]:
            row.append("(V)")
        else:
            row.append("   ")

        row.append(element_type_names[element_types[player]])
        row.append(web_names[player])
        row.append(total_points[player])

        data.append(row)

    print_table(data, headers)

    # Print the reserve players, including substitutions and captains
    print("Reserves:")

    headers = ["", "", "Position", "Name", "Points"]
    data = []
    for player in [
        substituted_roles["reserve_gkp"],
        substituted_roles["reserve_out_1"],
        substituted_roles["reserve_out_2"],
        substituted_roles["reserve_out_3"],
    ]:
        row = []

        if player in selected_roles["starting_xi"]:
            row.append("<-")
        else:
            row.append("  ")

        if player == selected_roles["captain"]:
            row.append("(*C)")
        elif player == selected_roles["vice_captain"]:
            row.append("(*V)")
        else:
            row.append("    ")

        row.append(element_type_names[element_types[player]])
        row.append(web_names[player])
        row.append(total_points[player])

        data.append(row)

    print_table(data, headers)

    # Print transfer activity and final budget
    print(f"Transfers ({transfer_cost} points) [Free transfers: {free_transfers}]")
    for player in set(final_squad) - set(initial_squad):
        print(f"-> {web_names[player]} ({format_currency(now_costs[player])})")
    for player in set(initial_squad) - set(final_squad):
        print(f"<- {web_names[player]} ({format_currency(selling_prices[player])})")

    print(f"Bank: {format_currency(final_budget)}")


def format_currency(amount: int):
    """Format game currency as a string."""
    return f"${round(amount / 10, 1)}"


def print_table(data, headers=None):
    """Prints a table from a list of lists."""
    if not data:
        return
    # Calculate column widths
    column_widths = [
        max(len(str(item)) for item in col)
        for col in zip(*(data + ([headers] if headers else [])), strict=False)
    ]
    # Print header
    if headers:
        print(
            " ".join(
                header.ljust(width)
                for header, width in zip(headers, column_widths, strict=False)
            )
        )
        print("-" * sum(column_widths + [len(headers) - 1]))
    # Print data rows
    for row in data:
        print(
            " ".join(
                str(item).ljust(width)
                for item, width in zip(row, column_widths, strict=False)
            )
        )