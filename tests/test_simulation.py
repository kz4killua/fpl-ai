import json

import polars as pl

from datautil.fpl import load_elements
from datautil.upcoming import remove_upcoming_data
from game.rules import DEF, FWD, GKP, MID
from simulation.simulate import Simulator
from simulation.utils import (
    calculate_points,
    calculate_selling_price,
    calculate_transfer_cost,
    load_results,
    make_automatic_substitutions,
)


def test_make_automatic_substitutions():
    # Entry 2267416 - 2023-24 season
    roles = {
        "captain": 415,
        "vice_captain": 352,
        "starting_xi": [352, 616, 206, 398, 31, 6, 501, 294, 303, 415, 297],
        "reserve_gkp": 597,
        "reserve_out_1": 368,
        "reserve_out_2": 209,
        "reserve_out_3": 278,
    }
    positions = {
        352: GKP,
        597: GKP,
        616: DEF,
        206: DEF,
        398: DEF,
        368: DEF,
        31: DEF,
        209: MID,
        6: MID,
        501: MID,
        294: MID,
        303: MID,
        415: FWD,
        297: FWD,
        278: FWD,
    }

    test_cases = [
        # Gameweek 1 - 1 substitution
        {
            "minutes": {
                352: 90,
                616: 11,
                206: 75,
                398: 90,
                209: 67,
                6: 90,
                501: 90,
                294: 65,
                303: 76,
                415: 67,
                297: 65,
                597: 90,
                368: 0,
                31: 0,
                278: 32,
            },
            "expected": {
                "captain": 415,
                "vice_captain": 352,
                "starting_xi": [352, 616, 206, 398, 209, 6, 501, 294, 303, 415, 297],
                "reserve_gkp": 597,
                "reserve_out_1": 368,
                "reserve_out_2": 31,
                "reserve_out_3": 278,
            },
        },
        # Gameweek 3 - 0 substitutions
        {
            "minutes": {
                352: 90,
                616: 90,
                206: 0,
                398: 0,
                209: 0,
                6: 55,
                501: 90,
                294: 32,
                303: 32,
                415: 71,
                297: 57,
                597: 90,
                368: 0,
                31: 34,
                278: 0,
            },
            "expected": {
                "captain": 415,
                "vice_captain": 352,
                "starting_xi": [352, 616, 206, 398, 31, 6, 501, 294, 303, 415, 297],
                "reserve_gkp": 597,
                "reserve_out_1": 368,
                "reserve_out_2": 209,
                "reserve_out_3": 278,
            },
        },
        # Gameweek 8 - 2 substitutions
        {
            "minutes": {
                352: 90,
                616: 90,
                206: 0,
                398: 0,
                209: 62,
                6: 15,
                501: 90,
                294: 0,
                303: 90,
                415: 85,
                297: 0,
                597: 90,
                368: 22,
                31: 74,
                278: 0,
            },
            "expected": {
                "captain": 415,
                "vice_captain": 352,
                "starting_xi": [352, 616, 368, 209, 31, 6, 501, 294, 303, 415, 297],
                "reserve_gkp": 597,
                "reserve_out_1": 206,
                "reserve_out_2": 398,
                "reserve_out_3": 278,
            },
        },
        # Gameweek 9 - 3 substitutions
        {
            "minutes": {
                352: 0,
                616: 90,
                206: 6,
                398: 0,
                209: 2,
                6: 12,
                501: 90,
                294: 90,
                303: 80,
                415: 20,
                297: 0,
                597: 90,
                368: 74,
                31: 45,
                278: 0,
            },
            "expected": {
                "captain": 415,
                "vice_captain": 352,
                "starting_xi": [597, 616, 206, 368, 31, 6, 501, 294, 303, 415, 209],
                "reserve_gkp": 352,
                "reserve_out_1": 398,
                "reserve_out_2": 297,
                "reserve_out_3": 278,
            },
        },
    ]

    for test_case in test_cases:
        substituted_roles = make_automatic_substitutions(
            roles, test_case["minutes"], positions
        )
        assert substituted_roles == test_case["expected"]

    # Entry 3291882 - 2024-25 season
    roles = {
        "captain": 755,
        "vice_captain": 328,
        "starting_xi": [201, 291, 311, 517, 182, 328, 74, 78, 755, 401, 110],
        "reserve_gkp": 109,
        "reserve_out_1": 770,
        "reserve_out_2": 270,
        "reserve_out_3": 457,
    }
    positions = {
        201: 1,
        291: 2,
        311: 2,
        270: 2,
        182: 3,
        328: 3,
        74: 3,
        78: 3,
        755: 4,
        401: 4,
        110: 4,
        109: 1,
        770: 3,
        517: 2,
        457: 2,
    }
    test_cases = [
        # Gameweek 33 - 1 substitution
        {
            "minutes": {
                201: 180,
                291: 90,
                311: 19,
                270: 90,
                182: 90,
                328: 90,
                74: 68,
                78: 90,
                755: 180,
                401: 75,
                110: 90,
                109: 0,
                770: 90,
                517: 0,
                457: 0,
            },
            "expected": {
                "captain": 755,
                "vice_captain": 328,
                "starting_xi": [201, 291, 311, 270, 182, 328, 74, 78, 755, 401, 110],
                "reserve_gkp": 109,
                "reserve_out_1": 770,
                "reserve_out_2": 517,
                "reserve_out_3": 457,
            },
        }
    ]

    for test_case in test_cases:
        substituted_roles = make_automatic_substitutions(
            roles, test_case["minutes"], positions
        )
        assert substituted_roles == test_case["expected"]


def test_remove_upcoming_data():
    elements = load_elements([2016, 2017])

    # Test removing data on or after the first gameweek
    filtered_elements = remove_upcoming_data(elements, 2017, 1)
    filtered_elements = filtered_elements.collect()
    assert filtered_elements.get_column("gameweek").max() == 38
    assert filtered_elements.get_column("season").max() == 2016

    # Test removing data from the middle of the season
    filtered_elements = remove_upcoming_data(elements, 2017, 20)
    filtered_elements = filtered_elements.collect()
    assert (
        filtered_elements.filter(pl.col("season") == 2016).get_column("gameweek").max()
        == 38
    )
    assert (
        filtered_elements.filter(pl.col("season") == 2017).get_column("gameweek").max()
        == 19
    )


def test_load_results():
    season = 2023
    results = load_results(season).collect()

    # Erling Haaland (355) scored 15 points in gameweek 37
    r1 = results.filter((pl.col("element") == 355) & (pl.col("gameweek") == 37))
    assert r1.select(["total_points", "minutes"]).to_dicts()[0] == {
        "total_points": 15,
        "minutes": 171,
    }

    # Josko Gvardiol (616) scored 27 points in gameweek 37
    r2 = results.filter((pl.col("element") == 616) & (pl.col("gameweek") == 37))
    assert r2.select(["total_points", "minutes"]).to_dicts()[0] == {
        "total_points": 27,
        "minutes": 180,
    }

    assert results.null_count().sum_horizontal().item() == 0


def test_calculate_points():
    roles = {
        "captain": 11,
        "vice_captain": 10,
        "starting_xi": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "reserve_gkp": 12,
        "reserve_out_1": 13,
        "reserve_out_2": 14,
        "reserve_out_3": 15,
    }
    total_points = {
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        8: 8,
        9: 9,
        10: 10,
        11: 11,
        12: 12,
        13: 13,
        14: 14,
        15: 15,
    }
    assert calculate_points(roles, total_points) == 77


def test_calculate_selling_price():
    assert calculate_selling_price(100, 106) == 103
    assert calculate_selling_price(100, 105) == 102
    assert calculate_selling_price(50, 54) == 52
    assert calculate_selling_price(50, 53) == 51
    assert calculate_selling_price(120, 117) == 117


def test_calculate_transfer_cost():
    free_transfers = 1
    transfers_made = 3
    next_gameweek = 15
    wildcard_gameweeks = [14, 25]
    assert (
        calculate_transfer_cost(
            free_transfers, transfers_made, next_gameweek, wildcard_gameweeks
        )
        == 8
    )

    free_transfers = 1
    transfers_made = 0
    next_gameweek = 15
    wildcard_gameweeks = [14, 25]
    assert (
        calculate_transfer_cost(
            free_transfers, transfers_made, next_gameweek, wildcard_gameweeks
        )
        == 0
    )

    free_transfers = 0
    transfers_made = 10
    next_gameweek = 1
    wildcard_gameweeks = [14, 25]
    assert (
        calculate_transfer_cost(
            free_transfers, transfers_made, next_gameweek, wildcard_gameweeks
        )
        == 0
    )

    free_transfers = 1
    transfers_made = 3
    next_gameweek = 14
    wildcard_gameweeks = [14, 25]
    assert (
        calculate_transfer_cost(
            free_transfers, transfers_made, next_gameweek, wildcard_gameweeks
        )
        == 0
    )


def test_simulator():
    season = 2024
    entry_id = 3291882

    # Load entry data
    with open(f"tests/data/simulation/{entry_id}_picks.json") as f:
        entry_picks = json.load(f)
    with open(f"tests/data/simulation/{entry_id}_history.json") as f:
        entry_history = json.load(f)
    wildcard_gameweeks = [
        chip["event"] for chip in entry_history["chips"] if chip["name"] == "wildcard"
    ]

    # Simulate the entry's season
    simulator = Simulator(season)
    while simulator.next_gameweek is not None:
        picks = entry_picks[str(simulator.next_gameweek)]
        roles = get_simulator_roles(picks)
        simulator.update(roles, wildcard_gameweeks, log=True)
        assert simulator.season_points == picks["entry_history"]["total_points"], (
            f"Total points mismatch in gameweek {simulator.next_gameweek}. "
            f"Expected {picks['entry_history']['total_points']}, "
            f"got {simulator.season_points}."
        )


def get_simulator_roles(picks: dict):
    """Convert the squad picks to simulator roles."""
    roles = {
        "captain": None,
        "vice_captain": None,
        "starting_xi": [],
        "reserve_gkp": None,
        "reserve_out_1": None,
        "reserve_out_2": None,
        "reserve_out_3": None,
    }

    for item in picks["picks"]:
        # Set the captain and vice-captain
        if item["is_captain"]:
            roles["captain"] = item["element"]
        elif item["is_vice_captain"]:
            roles["vice_captain"] = item["element"]

        # Set the starting XI and reserves
        if item["position"] >= 1 and item["position"] <= 11:
            roles["starting_xi"].append(item["element"])
        elif item["position"] == 12:
            roles["reserve_gkp"] = item["element"]
        elif item["position"] == 13:
            roles["reserve_out_1"] = item["element"]
        elif item["position"] == 14:
            roles["reserve_out_2"] = item["element"]
        elif item["position"] == 15:
            roles["reserve_out_3"] = item["element"]

    # Undo automatic substitutions
    for item in picks["automatic_subs"]:
        element_out = item["element_out"]
        element_in = item["element_in"]
        assert element_in in roles["starting_xi"]
        assert element_out in [
            roles["reserve_gkp"],
            roles["reserve_out_1"],
            roles["reserve_out_2"],
            roles["reserve_out_3"],
        ]

        for key in ["reserve_gkp", "reserve_out_1", "reserve_out_2", "reserve_out_3"]:
            if roles[key] == element_out:
                roles[key] = element_in
                idx = roles["starting_xi"].index(element_in)
                roles["starting_xi"].remove(element_in)
                roles["starting_xi"].insert(idx, element_out)
                break

    return roles
