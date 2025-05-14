import polars as pl
from polars.testing import assert_frame_equal

from datautil.load.fpl import load_fpl
from datautil.load.understat import load_understat


def test_load_fpl():
    # Load data for the 2016-17 season
    seasons = ["2016-17"]
    players, _ = load_fpl(seasons)
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
