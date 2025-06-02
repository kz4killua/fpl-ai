import polars as pl
from polars.testing import assert_frame_equal

from datautil.load.fpl import load_fixtures, load_fpl
from datautil.load.understat import load_understat
from datautil.upcoming import (
    get_upcoming_fixtures,
    get_upcoming_gameweeks,
)
from datautil.utils import get_seasons


def test_load_fpl():
    # Load data for the 2016-17 season
    seasons = ["2016-17"]
    players, _, _ = load_fpl(seasons)
    players = players.collect()

    # Test that mappings are correct
    expected = pl.DataFrame(
        {
            "season": ["2016-17", "2016-17", "2016-17", "2016-17"],
            "element": [73, 78, 12, 403],
            "fixture": [10, 10, 8, 3],
            "code": [60772, 19419, 37265, 78830],
            "element_type": [1, 2, 3, 4],
            "team_code": [8, 8, 3, 6],
            "opponent_team_code": [21, 21, 14, 11],
        },
    )
    result = players.join(
        expected.select(["season", "element", "fixture"]),
        on=["season", "element", "fixture"],
        how="inner",
    ).select(expected.columns)
    assert_frame_equal(
        result,
        expected,
        check_row_order=False,
        check_column_order=False,
        check_exact=True,
    )

    # Test that mappings leave no null values
    assert (
        players.select(
            [
                pl.col("code"),
                pl.col("element_type"),
                pl.col("team_code"),
                pl.col("opponent_team_code"),
            ]
        )
        .null_count()
        .sum_horizontal()
        .item()
        == 0
    )

    # Test that availability mappings are correct
    seasons = ["2024-25"]
    players, _, _ = load_fpl(seasons)
    players = players.collect()

    # Haaland (351) was injured between gameweeks 30 and 35
    expected = pl.DataFrame(
        {
            "season": ["2024-25", "2024-25", "2024-25", "2024-25"],
            "element": [351, 351, 351, 351],
            "round": [29, 30, 31, 36],
            "status": ["a", "i", "i", "a"],
        }
    )
    result = players.join(
        expected.select(["season", "element", "round"]),
        on=["season", "element", "round"],
        how="inner",
    ).select(expected.columns)
    assert_frame_equal(
        result,
        expected,
        check_row_order=False,
        check_column_order=False,
        check_exact=True,
    )


def test_load_understat():
    # Load data for the 2022-23 season
    seasons = ["2021-22"]
    players, teams = load_understat(seasons)
    players = players.collect()
    teams = teams.collect()

    # Test that the correct number of rows is returned
    assert players.filter(pl.col("id") == 453).height == 35
    assert players.filter(pl.col("id") == 447).height == 30
    assert teams.filter(pl.col("id") == 88).height == 38
    assert teams.filter(pl.col("id") == 82).height == 38

    # Test that player mappings are correct
    expected = pl.DataFrame(
        {
            "season": [2021, 2021],
            "id": [453, 447],
            "fpl_code": [85971, 61366],
            "fixture_id": [16385, 16385],
            "fpl_fixture_id": [10, 10],
        }
    )
    result = players.join(
        expected.select(["season", "id", "fpl_fixture_id"]),
        on=["season", "id", "fpl_fixture_id"],
        how="inner",
    ).select(expected.columns)
    assert_frame_equal(
        result,
        expected,
        check_row_order=False,
        check_column_order=False,
        check_exact=True,
        check_dtypes=False,
    )

    # Test that team mappings are correct
    expected = pl.DataFrame(
        {
            "season": [2021, 2021],
            "id": [88, 82],
            "fpl_code": [43, 6],
            "fixture_id": [16385, 16385],
            "fpl_fixture_id": [10, 10],
        }
    )
    result = teams.join(
        expected.select(["season", "id", "fpl_fixture_id"]),
        on=["season", "id", "fpl_fixture_id"],
        how="inner",
    ).select(expected.columns)
    assert_frame_equal(
        result,
        expected,
        check_row_order=False,
        check_column_order=False,
        check_exact=True,
        check_dtypes=False,
    )

    # Test that mappings leave no null values
    assert (
        players.select(
            [
                pl.col("fpl_code"),
                pl.col("fpl_fixture_id"),
            ]
        )
        .null_count()
        .sum_horizontal()
        .item()
        == 0
    )


def test_get_seasons():
    current_season = "2023-24"
    expected = [
        "2016-17",
        "2017-18",
        "2018-19",
        "2019-20",
        "2020-21",
        "2021-22",
        "2022-23",
        "2023-24",
    ]
    assert get_seasons(current_season) == expected

    expected = [
        "2021-22",
        "2022-23",
        "2023-24",
    ]
    assert get_seasons(current_season, 3) == expected


def test_get_upcoming_gameweeks():
    next_gameweek = 1
    last_gameweek = 38
    window_size = 5
    assert get_upcoming_gameweeks(
        next_gameweek,
        window_size,
        last_gameweek,
    ) == [1, 2, 3, 4, 5]

    next_gameweek = 38
    last_gameweek = 38
    window_size = 5
    assert get_upcoming_gameweeks(
        next_gameweek,
        window_size,
        last_gameweek,
    ) == [38]


def test_get_upcoming_fixtures():
    upcoming_gameweeks = [1, 2, 3, 4, 5]
    season = "2016-17"
    fixtures = load_fixtures([season])
    upcoming_fixtures = get_upcoming_fixtures(fixtures, season, upcoming_gameweeks)
    upcoming_fixtures = upcoming_fixtures.collect()
    assert upcoming_fixtures.height == 50
    assert upcoming_fixtures.null_count().sum_horizontal().item() == 0
    assert upcoming_fixtures.get_column("event").min() == 1
    assert upcoming_fixtures.get_column("event").max() == 5
