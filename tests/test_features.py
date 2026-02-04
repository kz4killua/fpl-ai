from datetime import datetime

import polars as pl
from polars.testing import assert_frame_equal

from features.availability import compute_availability
from features.balanced_mean import compute_balanced_mean
from features.fatigue import compute_fatigue
from features.imputed_last_season_mean import compute_imputed_last_season_mean
from features.imputed_set_piece_order import compute_imputed_set_piece_order
from features.last_season_mean import compute_last_season_mean
from features.minutes_category import compute_minutes_category
from features.per_90 import compute_per_90
from features.record_count import compute_record_count
from features.rolling_mean import compute_rolling_mean
from features.share import compute_share
from loaders.utils import force_dataframe


def test_compute_imputed_set_piece_order():
    df = pl.DataFrame(
        {
            "season": [2022, 2022, 2022],
            "gameweek": [1, 2, 3],
            "kickoff_time": [1, 2, 3],
            "element": [1, 1, 1],
            "penalties_order": [None, 1, None],
            "direct_freekicks_order": [None, 1, None],
            "corners_and_indirect_freekicks_order": [None, 1, None],
        }
    )
    expected = df.with_columns(
        pl.Series("penalties_order_missing", [1, 0, 0]),
        pl.Series("imputed_penalties_order", [11, 1, 1]),
        pl.Series("direct_freekicks_order_missing", [1, 0, 0]),
        pl.Series("imputed_direct_freekicks_order", [11, 1, 1]),
        pl.Series("corners_and_indirect_freekicks_order_missing", [1, 0, 0]),
        pl.Series("imputed_corners_and_indirect_freekicks_order", [11, 1, 1]),
    )
    result = compute_imputed_set_piece_order(df)
    assert_frame_equal(
        result,
        expected,
        check_row_order=False,
        check_column_order=False,
        check_exact=False,
        check_dtypes=False,
    )


def test_compute_fatigue():
    # Test with a single player
    df = pl.DataFrame(
        {
            "season": [2023] * 6,
            "kickoff_time": [
                datetime(2023, 8, 1, 14, 0),
                datetime(2023, 8, 5, 14, 0),
                datetime(2023, 8, 10, 14, 0),
                datetime(2023, 8, 15, 14, 0),
                datetime(2023, 8, 20, 14, 0),
                datetime(2023, 8, 25, 14, 0),
            ],
            "code": [1] * 6,
            "minutes": [90, 45, 60, None, 30, 90],
        }
    )
    expected = df.with_columns(
        pl.Series(
            "minutes_sum_10_days",
            [0, 90, 135, 105, 60, 30],
        )
    )
    df = df.sample(fraction=1.0, shuffle=True, seed=42)
    result = compute_fatigue(df, window=10)
    assert_frame_equal(
        result,
        expected,
        check_row_order=False,
        check_column_order=False,
        check_exact=False,
        check_dtypes=False,
    )

    # Test with multiple players
    df = pl.DataFrame(
        {
            "season": [2023] * 6 + [2023] * 6,
            "kickoff_time": [
                datetime(2023, 8, 1, 14, 0),
                datetime(2023, 8, 5, 14, 0),
                datetime(2023, 8, 10, 14, 0),
                datetime(2023, 8, 15, 14, 0),
                datetime(2023, 8, 20, 14, 0),
                datetime(2023, 8, 25, 14, 0),
            ]
            + [
                datetime(2023, 8, 2, 14, 0),
                datetime(2023, 8, 6, 14, 0),
                datetime(2023, 8, 11, 14, 0),
                datetime(2023, 8, 16, 14, 0),
                datetime(2023, 8, 21, 14, 0),
                datetime(2023, 8, 26, 14, 0),
            ],
            "code": [1] * 6 + [2] * 6,
            "minutes": [90, 45, 60, None, 30, 90] + [60, 60, None, 60, 60, 60],
        }
    )
    expected = df.with_columns(
        pl.Series(
            "minutes_sum_10_days",
            [0, 90, 135, 105, 60, 30] + [0, 60, 120, 60, 60, 120],
        )
    )
    df = df.sample(fraction=1.0, shuffle=True, seed=42)
    result = compute_fatigue(df, window=10)
    assert_frame_equal(
        result,
        expected,
        check_row_order=False,
        check_column_order=False,
        check_exact=False,
        check_dtypes=False,
    )


def test_compute_minutes_category():
    df = pl.DataFrame(
        {
            "minutes": [0, 1, 59, 60, 90, None],
        }
    )
    expected = df.with_columns(
        pl.Series(
            "minutes_category",
            [
                "0_minutes",
                "1_to_59_minutes",
                "1_to_59_minutes",
                "60_plus_minutes",
                "60_plus_minutes",
                None,
            ],
        )
    )
    result = compute_minutes_category(df)
    assert_frame_equal(result, expected)


def test_compute_record_count():
    # Test with a single player (no null values)
    df = pl.DataFrame(
        {
            "season": [2021] * 5,
            "kickoff_time": [1, 2, 3, 4, 5],
            "code": [1, 1, 1, 1, 1],
            "total_points": [0, 0, 0, 0, 0],
        }
    )
    expected = df.with_columns(pl.Series("record_count", [0, 1, 2, 3, 4]))
    result = compute_record_count(df, "total_points")
    assert_frame_equal(
        result,
        expected,
        check_row_order=False,
        check_column_order=False,
        check_dtypes=False,
    )

    # Test with null values
    df = pl.DataFrame(
        {
            "season": [2021] * 5,
            "kickoff_time": [1, 2, 3, 4, 5],
            "code": [1, 1, 1, 1, 1],
            "total_points": [0, 0, 0, None, None],
        }
    )
    expected = df.with_columns(pl.Series("record_count", [0, 1, 2, 3, 3]))
    result = compute_record_count(df, "total_points")
    assert_frame_equal(
        result,
        expected,
        check_row_order=False,
        check_column_order=False,
        check_dtypes=False,
    )

    # Test over multiple seasons
    df = pl.DataFrame(
        {
            "season": [2021] * 3 + [2022] * 3,
            "kickoff_time": [1, 2, 3, 4, 5, 6],
            "code": [1, 1, 1, 1, 1, 1],
            "total_points": [2, 2, 2, 2, 2, 2],
        }
    )
    expected = df.with_columns(pl.Series("record_count", [0, 1, 2, 0, 1, 2]))
    result = compute_record_count(df, "total_points")
    assert_frame_equal(
        result,
        expected,
        check_row_order=False,
        check_column_order=False,
        check_dtypes=False,
    )

    # Test with multiple players
    df = pl.DataFrame(
        {
            "season": [2021] * 6,
            "kickoff_time": [1, 1, 2, 2, 3, 3],
            "code": [1, 2, 1, 2, 1, 2],
            "total_points": [2, 2, 2, 2, None, None],
        }
    )
    expected = df.with_columns(pl.Series("record_count", [0, 0, 1, 1, 2, 2]))
    result = compute_record_count(df, "total_points")
    assert_frame_equal(
        result,
        expected,
        check_row_order=False,
        check_column_order=False,
        check_dtypes=False,
    )


def test_compute_share():
    # Test with a single team and fixture
    players = pl.DataFrame(
        {
            "season": [2021] * 5,
            "team_code": [1] * 5,
            "fixture": [1] * 5,
            "code": [1, 2, 3, 4, 5],
            "total_points": [6, 2, 5, 0, 9],
        }
    )
    expected = players.with_columns(
        pl.Series("total_points_share", [0.2727, 0.0909, 0.2273, 0.0, 0.4091])
    )
    result = compute_share(players, ["total_points"])
    assert_frame_equal(
        result,
        expected,
        check_row_order=False,
        check_column_order=False,
        check_exact=False,
        check_dtypes=False,
        atol=1e-3,
    )

    # Test with multiple fixtures
    players = pl.DataFrame(
        {
            "season": [2021] * 4,
            "team_code": [1, 1, 1, 1],
            "fixture": [1, 1, 2, 2],
            "code": [1, 2, 1, 2],
            "total_points": [6, 2, 5, 0],
        }
    )
    expected = players.with_columns(
        pl.Series("total_points_share", [0.75, 0.25, 1.0, 0.0])
    )
    result = compute_share(players, ["total_points"])
    assert_frame_equal(
        result,
        expected,
        check_row_order=False,
        check_column_order=False,
        check_exact=False,
        check_dtypes=False,
        atol=1e-3,
    )

    # Test zero division
    players = pl.DataFrame(
        {
            "season": [2021] * 5,
            "team_code": [1] * 5,
            "fixture": [1] * 5,
            "code": [1, 2, 3, 4, 5],
            "total_points": [0, 0, 0, 0, 0],
        }
    )
    expected = players.with_columns(
        pl.Series("total_points_share", [0.0, 0.0, 0.0, 0.0, 0.0])
    )
    result = compute_share(players, ["total_points"])
    assert_frame_equal(
        result,
        expected,
        check_row_order=False,
        check_column_order=False,
        check_exact=False,
        check_dtypes=False,
        atol=1e-3,
    )


def test_compute_rolling_mean():
    # Test rolling means with multiple players
    players = pl.DataFrame(
        {
            "season": [2021] * 14,
            "code": [1, 1, 1, 1, 1, 1, 1] + [2, 2, 2, 2, 2, 2, 2],
            "kickoff_time": [1, 2, 3, 4, 5, 6, 7] + [1, 2, 3, 4, 5, 6, 7],
            "total_points": [9, 5, 3, 9, 8, 6, 5] + [4, 6, 2, 3, 6, 9, 9],
        }
    )
    expected = players.with_columns(
        pl.Series(
            "total_points_rolling_mean_3",
            [
                None,
                9.0,
                7.0,
                17 / 3,
                17 / 3,
                20 / 3,
                23 / 3,
            ]
            + [
                None,
                4.0,
                5.0,
                4.0,
                11 / 3,
                11 / 3,
                18 / 3,
            ],
        ),
        pl.Series(
            "total_points_rolling_mean_5",
            [
                None,
                9.0,
                7.0,
                17 / 3,
                26 / 4,
                34 / 5,
                31 / 5,
            ]
            + [
                None,
                4.0,
                5.0,
                4.0,
                15 / 4,
                21 / 5,
                26 / 5,
            ],
        ),
    )
    players = players.sample(fraction=1.0, shuffle=True, seed=42)
    result = compute_rolling_mean(
        players,
        columns=["total_points", "total_points"],
        window_sizes=[3, 5],
    )
    assert_frame_equal(
        result,
        expected,
        check_row_order=False,
        check_column_order=False,
        check_exact=False,
        check_dtypes=False,
    )

    # Test rolling means with intermediate null values
    players = pl.DataFrame(
        {
            "season": [2021] * 14,
            "code": [1, 1, 1, 1, 1, 1, 1] + [2, 2, 2, 2, 2, 2, 2],
            "kickoff_time": [1, 2, 3, 4, 5, 6, 7] + [1, 2, 3, 4, 5, 6, 7],
            "total_points": [9, None, 3, 9, None, 6, 5] + [4, None, None, 3, 6, 9, 9],
        }
    )
    expected = players.with_columns(
        pl.Series(
            "total_points_rolling_mean_1",
            [None, 9, 9, 3, 9, 9, 6] + [None, 4, 4, 4, 3, 6, 9],
        )
    )
    players = players.sample(fraction=1.0, shuffle=True, seed=42)
    result = compute_rolling_mean(
        players,
        columns=["total_points"],
        window_sizes=[1],
    )
    assert_frame_equal(
        result,
        expected,
        check_row_order=False,
        check_column_order=False,
        check_exact=False,
        check_dtypes=False,
    )

    # Test rolling means with final null values
    players = pl.DataFrame(
        {
            "season": [2021] * 7,
            "code": [1, 1, 1, 1, 1, 1, 1],
            "kickoff_time": [1, 2, 3, 4, 5, 6, 7],
            "total_points": [9, 3, 3, None, None, None, None],
        }
    )
    expected = players.with_columns(
        pl.Series(
            "total_points_rolling_mean_3",
            [None, 9, 6, 5, 5, 5, 5],
        )
    )
    players = players.sample(fraction=1.0, shuffle=True, seed=42)
    result = compute_rolling_mean(
        players,
        columns=["total_points"],
        window_sizes=[3],
    )
    assert_frame_equal(
        result,
        expected,
        check_row_order=False,
        check_column_order=False,
        check_exact=False,
        check_dtypes=False,
    )

    # Test rolling means across multiple seasons
    players = pl.DataFrame(
        {
            "season": [2021] * 3 + [2022] * 3,
            "code": [1, 1, 1, 1, 1, 1],
            "kickoff_time": [1, 2, 3, 4, 5, 6],
            "total_points": [9, 3, 3, 1, 3, 2],
        }
    )
    expected = players.with_columns(
        pl.Series(
            "total_points_rolling_mean_3",
            [None, 9, 6, None, 1, 2],
        )
    )
    players = players.sample(fraction=1.0, shuffle=True, seed=42)
    result = compute_rolling_mean(
        players,
        columns=["total_points"],
        window_sizes=[3],
    )
    assert_frame_equal(
        result,
        expected,
        check_row_order=False,
        check_column_order=False,
        check_exact=False,
        check_dtypes=False,
    )

    # Test rolling means over multiple columns
    players = pl.DataFrame(
        {
            "season": [2021] * 7,
            "code": [1] * 7,
            "kickoff_time": [1, 2, 3, 4, 5, 6, 7],
            "goals_scored": [1, 2, None, 1, 3, 2, 1],
            "column_1": [1, 2, None, 4, None, 6, None],
            "column_2": [10, None, 30, 40, None, 60, None],
        }
    )
    expected = players.with_columns(
        pl.Series("column_1_rolling_mean_2", [None, 1.0, 1.5, 1.5, 3.0, 3.0, 5.0]),
        pl.Series(
            "column_2_rolling_mean_3", [None, 10.0, 10.0, 20.0, 80 / 3, 80 / 3, 130 / 3]
        ),
    )
    result = compute_rolling_mean(
        players,
        columns=["column_1", "column_2"],
        window_sizes=[2, 3],
    )
    assert_frame_equal(
        result,
        expected,
        check_row_order=False,
        check_column_order=False,
        check_exact=False,
        check_dtypes=False,
    )


def test_compute_last_season_mean():
    # Test with a single player
    players = pl.DataFrame(
        {
            "code": [1, 1, 1, 1],
            "gameweek": [1, 2, 1, 2],
            "season": [2021, 2021, 2022, 2022],
            "total_points": [9, 1, None, None],
        }
    )
    expected = players.with_columns(
        pl.Series(
            "total_points_mean_last_season",
            [None, None, 5, 5],
        )
    )
    players = players.sample(fraction=1.0, shuffle=True, seed=42)
    result = compute_last_season_mean(players, ["total_points"])
    assert_frame_equal(
        result,
        expected,
        check_row_order=False,
        check_column_order=False,
        check_exact=False,
        check_dtypes=False,
    )
    # Test with multiple players
    players = pl.DataFrame(
        {
            "code": [1, 2, 1, 2],
            "gameweek": [1, 1, 1, 1],
            "season": [2021, 2021, 2022, 2022],
            "total_points": [2, 5, 3, 7],
        }
    )
    expected = players.with_columns(
        pl.Series(
            "total_points_mean_last_season",
            [None, None, 2, 5],
        )
    )
    players = players.sample(fraction=1.0, shuffle=True, seed=42)
    result = compute_last_season_mean(players, ["total_points"])
    assert_frame_equal(
        result,
        expected,
        check_row_order=False,
        check_column_order=False,
        check_exact=False,
        check_dtypes=False,
    )


def test_compute_imputed_last_season_mean():
    players = pl.DataFrame(
        {
            "season": [
                2016,
                2017,
                2017,
                2017,
                2017,
                2017,
                2017,
            ],
            "gameweek": [1, 1, 1, 1, 1, 1, 1],
            "kickoff_time": [1, 2, 3, 4, 5, 6, 7],
            "element": [0, 1, 2, 3, 4, 5, 6],
            "element_type": [1, 1, 1, 1, 4, 4, 4],
            "value": [
                50,
                50,
                100,
                75,
                100,
                200,
                150,
            ],
            "goals_scored_mean_last_season": [
                None,
                0.0,
                10.0,
                None,
                10.0,
                20.0,
                None,
            ],
        }
    )
    expected = players.with_columns(
        pl.Series(
            "imputed_goals_scored_mean_last_season",
            [
                None,
                0.0,
                10.0,
                5.0,  # 0.2 * 75 - 10 = 5.0
                10.0,
                20.0,
                15.0,  # 0.1 * 150 + 0 = 15.0
            ],
        )
    )
    result = compute_imputed_last_season_mean(players, "goals_scored_mean_last_season")
    assert_frame_equal(
        force_dataframe(result),
        expected,
        check_row_order=False,
        check_column_order=False,
        check_exact=False,
        check_dtypes=False,
    )


def test_compute_balanced_mean():
    # Test without null values
    players = pl.DataFrame(
        {
            "total_points_rolling_mean_5": [2, 8, 5],
            "total_points_mean_last_season": [4, 4, 4],
            "record_count": [0, 1, 2],
        }
    )
    expected = players.with_columns(
        pl.Series("balanced_total_points_rolling_mean_5", [4.0, 6.0, 4.75])
    )
    result = compute_balanced_mean(
        players,
        "total_points_rolling_mean_5",
        "total_points_mean_last_season",
        decay=0.5,
        default=0,
    )
    assert_frame_equal(
        result,
        expected,
        check_row_order=False,
        check_column_order=False,
        check_exact=False,
        check_dtypes=False,
    )

    # Test with null values
    players = pl.DataFrame(
        {
            "total_points_rolling_mean_5": [None, 12, 2],
            "total_points_mean_last_season": [None, None, None],
            "record_count": [0, 1, 2],
        }
    )
    expected = players.with_columns(
        pl.Series("balanced_total_points_rolling_mean_5", [0.0, 6.0, 1.5])
    )
    result = compute_balanced_mean(
        players,
        "total_points_rolling_mean_5",
        "total_points_mean_last_season",
        decay=0.5,
        default=0,
    )
    assert_frame_equal(
        result,
        expected,
        check_row_order=False,
        check_column_order=False,
        check_exact=False,
        check_dtypes=False,
    )


def test_compute_per_90():
    players = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "minutes": [0, 90, 90, 45, 90, None],
            "goals": [0, 0, 2, 2, 2, None],
        }
    )
    expected = pl.DataFrame(
        {
            "goals_per_90": [None, 0, 2, 4, 2, None],
        }
    )
    result = compute_per_90(players, ["goals"])
    result = result.select(expected.columns)
    assert_frame_equal(result, expected, check_dtypes=False)


def test_compute_availability():
    df = pl.DataFrame(
        {
            "season": [2019, 2019, 2019],
            "gameweek": [1, 2, 3],
            "code": [1, 1, 1],
            "kickoff_time": [
                datetime(2019, 8, 10, 15, 0),
                datetime(2019, 8, 17, 15, 0),
                datetime(2019, 8, 24, 15, 0),
            ],
        }
    )

    # Test with status 'a' (available)
    df = df.with_columns(
        [
            pl.Series("chance_of_playing_next_round", [None, None, None]),
            pl.Series("status", ["a", None, None]),
            pl.Series("news", ["", None, None]),
            pl.Series("news_added", [None, None, None]),
        ]
    )
    expected = pl.DataFrame({"availability": [100, 100, 100]})
    _compare_availability(df, expected)

    df = df.with_columns(
        [
            pl.Series("chance_of_playing_next_round", [100, None, None]),
            pl.Series("status", ["a", None, None]),
            pl.Series("news", ["", None, None]),
            pl.Series("news_added", [datetime(2019, 8, 1), None, None]),
        ]
    )
    expected = pl.DataFrame({"availability": [100, 100, 100]})
    _compare_availability(df, expected)

    # Test with status 'i' (injured)
    df = df.with_columns(
        [
            pl.Series("status", ["i", None, None]),
            pl.Series("news", ["Ankle injury - Expected back 15 Aug", None, None]),
            pl.Series("news_added", [datetime(2019, 8, 1), None, None]),
            pl.Series("chance_of_playing_next_round", [None, None, None]),
        ]
    )
    expected = pl.DataFrame({"availability": [0, 100, 100]})
    _compare_availability(df, expected)

    df = df.with_columns(
        [
            pl.Series("status", ["i", None, None]),
            pl.Series("news", ["Ankle injury - Expected back 15 Feb", None, None]),
            pl.Series("news_added", [datetime(2019, 8, 1), None, None]),
            pl.Series("chance_of_playing_next_round", [None, None, None]),
        ]
    )
    expected = pl.DataFrame({"availability": [0, 0, 0]})
    _compare_availability(df, expected)

    df = df.with_columns(
        [
            pl.Series("status", ["i", None, None]),
            pl.Series("news", ["Knee injury - Unknown return date", None, None]),
            pl.Series("news_added", [datetime(2019, 8, 1), None, None]),
            pl.Series("chance_of_playing_next_round", [None, None, None]),
        ]
    )
    expected = pl.DataFrame({"availability": [0, 0, 0]})
    _compare_availability(df, expected)

    # Test with status 'd' (doubtful)
    df = df.with_columns(
        [
            pl.Series("status", ["d", None, None]),
            pl.Series("news", ["Knock - 75% chance of playing", None, None]),
            pl.Series("news_added", [datetime(2019, 8, 1), None, None]),
            pl.Series("chance_of_playing_next_round", [75, None, None]),
        ]
    )
    expected = pl.DataFrame({"availability": [75, 100, 100]})
    _compare_availability(df, expected)

    # Test with status 's' (suspended)
    df = df.with_columns(
        [
            pl.Series("status", ["s", None, None]),
            pl.Series("news", ["Suspended until 20 Aug", None, None]),
            pl.Series("news_added", [datetime(2019, 8, 1), None, None]),
            pl.Series("chance_of_playing_next_round", [0, None, None]),
        ]
    )
    expected = pl.DataFrame({"availability": [0, 0, 100]})
    _compare_availability(df, expected)

    df = df.with_columns(
        [
            pl.Series("status", ["s", None, None]),
            pl.Series("news", ["Suspended until 20 Feb", None, None]),
            pl.Series("news_added", [datetime(2019, 8, 1), None, None]),
            pl.Series("chance_of_playing_next_round", [0, None, None]),
        ]
    )
    expected = pl.DataFrame({"availability": [0, 0, 0]})
    _compare_availability(df, expected)

    # Test with status 'u' (unavailable)
    df = df.with_columns(
        [
            pl.Series("status", ["u", None, None]),
            pl.Series("news", ["Transferred to Royal Antwerp", None, None]),
            pl.Series("news_added", [datetime(2019, 8, 1), None, None]),
            pl.Series("chance_of_playing_next_round", [0, None, None]),
        ]
    )
    expected = pl.DataFrame({"availability": [0, 0, 0]})
    _compare_availability(df, expected)

    df = df.with_columns(
        [
            pl.Series("status", ["u", None, None]),
            pl.Series(
                "news",
                [
                    "Joined Marseille on loan for 2021/22. - Expected back 20 Aug",
                    None,
                    None,
                ],
            ),
            pl.Series("news_added", [datetime(2019, 8, 1), None, None]),
            pl.Series("chance_of_playing_next_round", [0, None, None]),
        ]
    )
    expected = pl.DataFrame({"availability": [0, 0, 100]})
    _compare_availability(df, expected)

    df = df.with_columns(
        [
            pl.Series("status", ["u", None, None]),
            pl.Series(
                "news",
                [
                    "Joined Marseille on loan for 2021/22. - Expected back 20 Feb",
                    None,
                    None,
                ],
            ),
            pl.Series("news_added", [datetime(2019, 8, 1), None, None]),
            pl.Series("chance_of_playing_next_round", [0, None, None]),
        ]
    )
    expected = pl.DataFrame({"availability": [0, 0, 0]})
    _compare_availability(df, expected)

    # Test with status 'n' (not eligible)
    df = df.with_columns(
        pl.Series("status", ["n", None, None]),
        pl.Series(
            "news",
            [
                "Ineligible to face his parent club on 19/5. - Expected back 20 Aug",
                None,
                None,
            ],
        ),
        pl.Series("news_added", [datetime(2019, 8, 1), None, None]),
        pl.Series("chance_of_playing_next_round", [0, None, None]),
    )
    expected = pl.DataFrame({"availability": [0, 0, 100]})
    _compare_availability(df, expected)

    df = df.with_columns(
        pl.Series("status", ["n", None, None]),
        pl.Series(
            "news",
            [
                "Ineligible to face his parent club on 19/5. - Expected back 20 Feb",
                None,
                None,
            ],
        ),
        pl.Series("news_added", [datetime(2019, 8, 1), None, None]),
        pl.Series("chance_of_playing_next_round", [0, None, None]),
    )
    expected = pl.DataFrame({"availability": [0, 0, 0]})
    _compare_availability(df, expected)

    df = df.with_columns(
        [
            pl.Series("status", ["n", None, None]),
            pl.Series(
                "news", ["Transferred to Udinese - Unknown return date", None, None]
            ),
            pl.Series("news_added", [datetime(2019, 8, 1), None, None]),
            pl.Series("chance_of_playing_next_round", [0, None, None]),
        ]
    )
    expected = pl.DataFrame({"availability": [0, 0, 0]})
    _compare_availability(df, expected)

    # Test with multiple players
    df = pl.DataFrame(
        {
            "season": [2019, 2019, 2019, 2019],
            "gameweek": [1, 1, 2, 2],
            "code": [1, 2, 1, 2],
            "kickoff_time": [
                datetime(2019, 8, 10, 15, 0),
                datetime(2019, 8, 10, 15, 0),
                datetime(2019, 8, 17, 15, 0),
                datetime(2019, 8, 17, 15, 0),
            ],
        }
    )

    df = df.with_columns(
        [
            pl.Series("chance_of_playing_next_round", [None, 75, None, None]),
            pl.Series("status", ["a", "u", None, None]),
            pl.Series(
                "news",
                [
                    "",
                    "Transferred to Royal Antwerp",
                    None,
                    None,
                ],
            ),
            pl.Series(
                "news_added",
                [
                    datetime(2019, 8, 1),
                    datetime(2019, 8, 1),
                    None,
                    None,
                ],
            ),
        ]
    )
    expected = pl.DataFrame({"availability": [100, 0, 100, 0]})
    _compare_availability(df, expected)

    # Test with null availability
    df = df.with_columns(
        [
            pl.Series("chance_of_playing_next_round", [None, None, None, 100]),
            pl.Series("status", [None, None, None, "a"]),
            pl.Series("news", [None, None, None, ""]),
        ]
    )
    expected = pl.DataFrame({"availability": [None, None, None, 100]})
    _compare_availability(df, expected)


def _compare_availability(df: pl.DataFrame, expected: pl.DataFrame):
    df = df.with_columns(pl.col("news_added").cast(pl.Datetime))
    result = compute_availability(df)
    result = result.select(expected.columns)
    assert_frame_equal(result, expected, check_dtypes=False)
