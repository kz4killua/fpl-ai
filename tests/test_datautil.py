from datetime import datetime

import polars as pl
from polars.testing import assert_frame_equal

from datautil.load.clubelo import load_clubelo
from datautil.load.fpl import (
    load_fixtures,
    load_fpl,
)
from datautil.load.merged import load_merged
from datautil.load.understat import load_understat
from datautil.upcoming import (
    get_upcoming_fixtures,
    get_upcoming_gameweeks,
)
from datautil.utils import get_seasons


def test_load_clubelo():
    # Load clubelo ratings
    df = load_clubelo(datetime.max)
    df = df.collect()

    # Check that FPL team codes are mapped correctly
    expected = pl.DataFrame(
        {
            "Club": ["Arsenal", "Chelsea", "Liverpool"],
            "fpl_code": [3, 8, 14],
        }
    )
    assert_mappings_correct(
        df,
        expected,
        on=["Club"],
    )

    # Check that no null values are present in the fpl_code column
    assert df.get_column("fpl_code").null_count() == 0


def test_load_fpl():
    # Load FPL data
    seasons = ["2016-17", "2024-25"]
    players, teams, managers = load_fpl(seasons)
    players = players.collect()
    teams = teams.collect()
    managers = managers.collect()

    # Test mappings for player element types and teams
    expected = pl.DataFrame(
        {
            "season": ["2016-17", "2016-17", "2016-17", "2016-17"],
            "element": [73, 78, 12, 403],
            "fixture": [10, 10, 8, 3],
            "code": [60772, 19419, 37265, 78830],
            "element_type": [1, 2, 3, 4],
            "team": [4, 4, 1, 17],
            "team_code": [8, 8, 3, 6],
            "opponent_team": [20, 20, 9, 6],
            "opponent_team_code": [21, 21, 14, 11],
        },
    )
    assert_mappings_correct(
        players,
        expected,
        on=["season", "element", "fixture"],
    )

    # Test for null player element types and teams
    assert (
        players.select(
            [
                pl.col("code"),
                pl.col("element_type"),
                pl.col("team"),
                pl.col("team_code"),
                pl.col("opponent_team"),
                pl.col("opponent_team_code"),
            ]
        )
        .null_count()
        .sum_horizontal()
        .item()
        == 0
    )

    # Test mappings for player availability
    expected = pl.DataFrame(
        {
            "season": ["2024-25", "2024-25", "2024-25", "2024-25"],
            "element": [351, 351, 351, 351],
            "fixture": [288, 298, 308, 358],
            "round": [29, 30, 31, 36],
            "status": ["a", "i", "i", "a"],
        }
    )
    assert_mappings_correct(
        players,
        expected,
        on=["season", "element", "fixture"],
    )

    # Test that the correct number of rows (for teams) is returned
    assert teams.height == 20 * 38 * 2

    # Test mappings for team match details
    expected = pl.DataFrame(
        {
            "season": ["2024-25", "2024-25", "2024-25", "2024-25"],
            "round": [1, 2, 3, 4],
            "id": [13, 13, 13, 13],
            "code": [43, 43, 43, 43],
            "opponent_id": [6, 10, 19, 4],
            "opponent_code": [8, 40, 21, 94],
            "was_home": [False, True, False, True],
            "scored": [2, 4, 3, 2],
            "conceded": [0, 1, 1, 1],
        }
    )
    assert_mappings_correct(
        teams,
        expected,
        on=["season", "round", "id"],
    )

    # Test mappings for team codes across seasons
    expected = pl.DataFrame(
        {
            "season": ["2016-17", "2016-17", "2024-25", "2024-25"],
            "id": [10, 20, 10, 20],
            "code": [43, 21, 40, 39],
        }
    )
    assert_mappings_correct(
        teams,
        expected,
        on=["season", "id"],
    )

    # Test mappings for team strengths
    expected = pl.DataFrame(
        {
            "season": ["2024-25", "2024-25", "2024-25", "2024-25"],
            "id": [13, 13, 13, 13],
            "round": [1, 11, 21, 31],
            "strength_overall_home": [1355, 1355, 1220, 1230],
            "strength_overall_away": [1380, 1370, 1290, 1250],
        }
    )
    assert_mappings_correct(teams, expected, on=["season", "id", "round"])

    # Test that the correct number of managers is returned
    assert (
        len(
            managers.filter(pl.col("season") == "2024-25")
            .get_column("element")
            .unique()
            .to_list()
        )
        == 20
    )


def test_load_understat():
    # Load data for the 2021-22 season
    seasons = ["2021-22"]
    players, teams = load_understat(seasons, datetime.max)
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
            "fpl_season": ["2021-22", "2021-22"],
            "id": [453, 447],
            "fpl_code": [85971, 61366],
            "fixture_id": [16385, 16385],
            "fpl_fixture_id": [10, 10],
        }
    )
    assert_mappings_correct(
        players,
        expected,
        on=["season", "id", "fpl_fixture_id"],
    )

    # Test that team mappings are correct
    expected = pl.DataFrame(
        {
            "season": [2021, 2021],
            "fpl_season": ["2021-22", "2021-22"],
            "id": [88, 82],
            "fpl_code": [43, 6],
            "fixture_id": [16385, 16385],
            "fpl_fixture_id": [10, 10],
        }
    )
    assert_mappings_correct(
        teams,
        expected,
        on=["season", "id", "fpl_fixture_id"],
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
    assert (
        teams.select(
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


def test_load_merged():
    # Load data, including for upcoming fixtures
    seasons = ["2024-25", "2025-26"]
    current_season = "2025-26"
    upcoming_gameweeks = [1, 2, 3]
    players, teams, managers = load_merged(seasons, current_season, upcoming_gameweeks)
    players = players.collect()
    teams = teams.collect()
    managers = managers.collect()

    # Test non-upcoming player mappings
    expected = pl.DataFrame(
        {
            "season": ["2024-25", "2024-25", "2024-25", "2024-25"],
            "element": [351, 351, 351, 351],
            "round": [1, 2, 3, 4],
            "uds_xG": [0.66, 1.84, 1.31, 0.73],
            "uds_xA": [0.00, 0.00, 0.41, 0.00],
        }
    )
    assert_mappings_correct(
        players,
        expected,
        on=["season", "element", "round"],
        atol=1e-2,
    )

    # Test non-upcoming team mappings
    expected = pl.DataFrame(
        {
            "season": ["2024-25", "2024-25", "2024-25", "2024-25"],
            "code": [43, 43, 43, 43],
            "id": [13, 13, 13, 13],
            "round": [1, 2, 3, 4],
            "uds_xG": [1.18, 3.08, 3.19, 1.55],
            "uds_xGA": [1.06, 0.48, 0.95, 1.05],
            "clb_elo": [2050.57299805, 2055.97094727, 2056.84448242, 2060.20605469],
        }
    )
    assert_mappings_correct(
        teams,
        expected,
        on=["season", "code", "round"],
        atol=1e-2,
    )

    # Test upcoming player mappings
    expected = pl.DataFrame(
        {
            "season": ["2024-25", "2025-26", "2025-26", "2025-26"],
            "code": [118748, 118748, 118748, 118748],
            "round": [38, 1, 2, 3],
            "element": [328, 381, 381, 381],
            "value": [136, 145, 145, 145],
            "team": [12, 12, 12, 12],
            "opponent_team": [7, 4, 15, 1],
            "was_home": [1, 1, 0, 1],
            # Availability should be only be filled for the next gameweek
            "status": ["a", "a", None, None],
            "uds_xG": [0.65, None, None, None],
        }
    )
    assert_mappings_correct(
        players,
        expected,
        on=["season", "code", "round"],
        atol=1e-2,
    )

    # Test upcoming team mappings
    expected = pl.DataFrame(
        {
            "season": ["2024-25", "2025-26", "2025-26", "2025-26"],
            "round": [38, 1, 2, 3],
            "code": [14, 14, 14, 14],
            "id": [12, 12, 12, 12],
            "opponent_id": [7, 4, 15, 1],
            "was_home": [1, 1, 0, 1],
            "uds_xG": [1.92, None, None, None],
            "uds_xGA": [1.41, None, None, None],
            "clb_elo": [1996.71533203, 1993.41772461, 1993.41772461, 1993.41772461],
        }
    )
    assert_mappings_correct(
        teams,
        expected,
        on=["season", "code", "round"],
        atol=1e-2,
    )

    # Test that the correct number of rows (for upcoming players) is returned
    assert players.filter(
        pl.col("season").eq(current_season)
        & pl.col("round").is_in(upcoming_gameweeks)
        & pl.col("code").eq(118748)
    ).height == len(upcoming_gameweeks)

    # Test that the correct number of rows (for upcoming teams) is returned
    assert teams.filter(
        pl.col("season").eq(current_season)
        & pl.col("round").is_in(upcoming_gameweeks)
        & pl.col("code").eq(14)
    ).height == len(upcoming_gameweeks)

    # Test for null values in upcoming players
    assert players.select("team", "team_code").null_count().sum_horizontal().item() == 0

    # Test for null values in team clubelo ratings
    assert teams.get_column("clb_elo").null_count() == 0


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
    assert get_upcoming_gameweeks(1, 5, 83) == [1, 2, 3, 4, 5]
    assert get_upcoming_gameweeks(38, 5, 38) == [38]


def test_get_upcoming_fixtures():
    upcoming_gameweeks = [1, 2, 3, 4, 5]
    season = "2016-17"
    fixtures = load_fixtures([season])
    upcoming_fixtures = get_upcoming_fixtures(fixtures, season, upcoming_gameweeks)
    upcoming_fixtures = upcoming_fixtures.collect()
    assert upcoming_fixtures.height == 50
    assert upcoming_fixtures.get_column("event").min() == 1
    assert upcoming_fixtures.get_column("event").max() == 5


def assert_mappings_correct(
    df: pl.DataFrame,
    expected: pl.DataFrame,
    on: list[str],
    *,
    check_row_order: bool = False,
    check_column_order: bool = False,
    check_exact: bool = False,
    check_dtypes: bool = False,
    atol: float = 1e-8,
    rtol: float = 1e-5,
):
    result = (
        df.join(
            expected.select(on).unique(),
            on=on,
            how="inner",
        )
        .select(expected.columns)
        .unique()
    )
    assert_frame_equal(
        result,
        expected,
        check_row_order=check_row_order,
        check_column_order=check_column_order,
        check_exact=check_exact,
        check_dtypes=check_dtypes,
        atol=atol,
        rtol=rtol,
    )
