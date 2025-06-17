import polars as pl
from polars.testing import assert_frame_equal

from datautil.load.fpl import load_fixtures, load_fpl
from datautil.load.fplcache import load_static_elements, load_static_teams
from datautil.load.merged import load_merged
from datautil.load.understat import load_understat
from datautil.upcoming import (
    get_upcoming_fixtures,
    get_upcoming_gameweeks,
    get_upcoming_manager_data,
    get_upcoming_player_data,
    get_upcoming_team_data,
)
from datautil.utils import get_seasons
from game.rules import DEF, FWD, GKP, MID, MNG


def test_load_fpl():
    seasons = ["2016-17", "2024-25"]
    players, teams, managers = load_fpl(seasons)
    players = players.collect()
    teams = teams.collect()
    managers = managers.collect()
    _test_load_fpl_players(players)
    _test_load_fpl_teams(teams)
    _test_load_fpl_managers(managers)


def _test_load_fpl_players(players: pl.DataFrame):
    # Test mappings for element types and teams
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

    # Test for null element types and teams
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

    # Test mappings for availability
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


def _test_load_fpl_teams(teams: pl.DataFrame):
    # Test that the correct number of rows is returned
    assert teams.height == 20 * 38 * 2

    # Test mappings for match details
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


def _test_load_fpl_managers(managers: pl.DataFrame):
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
    players, teams, managers = load_merged(["2024-25"])
    players = players.collect()
    teams = teams.collect()
    managers = managers.collect()

    # Test FPL to understat player mappings
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

    # Test FPL to understat team mappings
    expected = pl.DataFrame(
        {
            "season": ["2024-25", "2024-25", "2024-25", "2024-25"],
            "id": [13, 13, 13, 13],
            "round": [1, 2, 3, 4],
            "uds_xG": [1.18, 3.08, 3.19, 1.55],
            "uds_xGA": [1.06, 0.48, 0.95, 1.05],
        }
    )
    assert_mappings_correct(
        teams,
        expected,
        on=["season", "id", "round"],
        atol=1e-2,
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


def test_get_upcoming_player_data():
    season = "2024-25"
    next_gameweek = 1
    upcoming_gameweeks = get_upcoming_gameweeks(next_gameweek, 5, 38)
    fixtures = load_fixtures([season])
    static_elements = load_static_elements(season, next_gameweek)
    static_teams = load_static_teams(season, next_gameweek)
    static_players = static_elements.filter(
        pl.col("element_type").is_in([GKP, DEF, MID, FWD])
    )
    upcoming_fixtures = get_upcoming_fixtures(fixtures, season, upcoming_gameweeks)
    upcoming_players = get_upcoming_player_data(
        upcoming_fixtures, static_players, static_teams
    )
    upcoming_players = upcoming_players.collect()

    # Test that mappings are correct
    expected = pl.DataFrame(
        {
            "season": ["2024-25", "2024-25", "2024-25", "2024-25"],
            "element": [351, 351, 351, 351],
            "round": [1, 2, 3, 4],
            "element_type": [FWD, FWD, FWD, FWD],
            "code": [223094, 223094, 223094, 223094],
            "team": [13, 13, 13, 13],
            "team_code": [43, 43, 43, 43],
            "opponent_team": [6, 10, 19, 4],
            "opponent_team_code": [8, 40, 21, 94],
            "was_home": [False, True, False, True],
            # Availability should be filled for (only) the next gameweek
            "status": ["a", None, None, None],
        }
    )
    assert_mappings_correct(
        upcoming_players,
        expected,
        on=["season", "element", "round"],
    )


def test_get_upcoming_manager_data():
    season = "2024-25"
    next_gameweek = 24
    upcoming_gameweeks = get_upcoming_gameweeks(next_gameweek, 5, 38)
    fixtures = load_fixtures([season])
    static_elements = load_static_elements(season, next_gameweek)
    static_teams = load_static_teams(season, next_gameweek)
    static_managers = static_elements.filter(pl.col("element_type") == MNG)
    upcoming_fixtures = get_upcoming_fixtures(fixtures, season, upcoming_gameweeks)
    upcoming_managers = get_upcoming_manager_data(
        upcoming_fixtures, static_managers, static_teams
    )
    upcoming_managers = upcoming_managers.collect()

    # Test that the correct number of rows is returned
    assert upcoming_managers.height == 22 + 22 + 20 + 20 + 20

    # Test that mappings are correct
    expected = pl.DataFrame(
        {
            "season": ["2024-25", "2024-25", "2024-25", "2024-25"],
            "element": [736, 736, 736, 736],
            "round": [24, 25, 26, 27],
            "element_type": [MNG, MNG, MNG, MNG],
            "code": [100037973, 100037973, 100037973, 100037973],
            "team": [13, 13, 13, 13],
            "team_code": [43, 43, 43, 43],
            "opponent_team": [1, 15, 12, 18],
            "opponent_team_code": [3, 4, 14, 6],
            "was_home": [False, True, True, False],
        }
    )
    assert_mappings_correct(
        upcoming_managers,
        expected,
        on=["season", "element", "round"],
    )


def test_get_upcoming_team_data():
    season = "2024-25"
    next_gameweek = 1
    static_teams = load_static_teams(season, next_gameweek)
    fixtures = load_fixtures([season])
    upcoming_gameweeks = get_upcoming_gameweeks(next_gameweek, 5, 38)
    upcoming_fixtures = get_upcoming_fixtures(fixtures, season, upcoming_gameweeks)
    upcoming_teams = get_upcoming_team_data(upcoming_fixtures, static_teams)
    upcoming_teams = upcoming_teams.collect()

    # Test that the correct number of rows is returned
    assert upcoming_teams.height == 20 * 5

    # Test that mappings are correct
    expected = pl.DataFrame(
        {
            "season": ["2024-25", "2024-25", "2024-25", "2024-25"],
            "id": [13, 13, 13, 13],
            "round": [1, 2, 3, 4],
            "code": [43, 43, 43, 43],
            "opponent_id": [6, 10, 19, 4],
            "opponent_code": [8, 40, 21, 94],
            "was_home": [False, True, False, True],
            "strength_overall_home": [1355, 1355, 1355, 1355],
            "strength_overall_away": [1380, 1380, 1380, 1380],
        }
    )
    assert_mappings_correct(
        upcoming_teams,
        expected,
        on=["season", "id", "round"],
    )


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
