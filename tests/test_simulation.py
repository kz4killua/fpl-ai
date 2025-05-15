import polars as pl

from datautil.load.fpl import load_elements
from optimization.rules import DEF, FWD, GKP, MID
from simulation.utils import (
    calculate_points,
    load_results,
    make_automatic_substitutions,
    remove_upcoming_data,
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
    elements = load_elements(["2016-17", "2017-18"])

    # Test removing data on or after the first gameweek
    filtered_elements = remove_upcoming_data(elements, "2017-18", 1)
    filtered_elements = filtered_elements.collect()
    assert filtered_elements.get_column("round").max() == 38
    assert filtered_elements.get_column("season").max() == "2016-17"

    # Test removing data from the middle of the season
    filtered_elements = remove_upcoming_data(elements, "2017-18", 20)
    filtered_elements = filtered_elements.collect()
    assert (
        filtered_elements.filter(pl.col("season") == "2016-17")
        .get_column("round")
        .max()
        == 38
    )
    assert (
        filtered_elements.filter(pl.col("season") == "2017-18")
        .get_column("round")
        .max()
        == 19
    )


def test_load_results():
    season = "2023-24"
    results = load_results(season).collect()

    # Erling Haaland (355) scored 15 points in gameweek 37
    r1 = results.filter((pl.col("element") == 355) & (pl.col("round") == 37))
    assert r1.select(["total_points", "minutes"]).to_dicts()[0] == {
        "total_points": 15,
        "minutes": 171,
    }

    # Josko Gvardiol (616) scored 27 points in gameweek 37
    r2 = results.filter((pl.col("element") == 616) & (pl.col("round") == 37))
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
