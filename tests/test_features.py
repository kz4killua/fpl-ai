from datetime import datetime

import polars as pl
from polars.testing import assert_frame_equal

from features.availability import compute_availability
from features.rolling_mean import compute_rolling_mean
from features.rolling_sum_over_days import rolling_sum_over_days


def test_compute_rolling_mean():
    # Test rolling means with multiple players
    players = pl.DataFrame(
        {
            "id": [1, 1, 1, 1, 1, 1, 1] + [2, 2, 2, 2, 2, 2, 2],
            "round": [1, 2, 3, 4, 5, 6, 7] + [1, 2, 3, 4, 5, 6, 7],
            "total_points": [9, 5, 3, 9, 8, 6, 5] + [4, 6, 2, 3, 6, 9, 9],
        }
    )
    expected = players.with_columns(
        pl.Series(
            "total_points_rolling_mean_3",
            [
                0.0,
                9.0,
                7.0,
                17 / 3,
                17 / 3,
                20 / 3,
                23 / 3,
            ]
            + [
                0.0,
                4.0,
                5.0,
                4.0,
                11 / 3,
                11 / 3,
                18 / 3,
            ]
        ),
        pl.Series(
            "total_points_rolling_mean_5",
            [
                0.0,
                9.0,
                7.0,
                17 / 3,
                26 / 4,
                34 / 5,
                31 / 5,
            ]
            + [
                0.0,
                4.0,
                5.0,
                4.0,
                15 / 4,
                21 / 5,
                26 / 5,
            ]
        )
    )
    players = players.sample(fraction=1.0, shuffle=True, seed=42)
    result = compute_rolling_mean(
        players,
        order_by="round",
        group_by="id",
        columns=["total_points", "total_points"],
        window_sizes=[3, 5],
        defaults=[0, 0],
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
            "id": [1, 1, 1, 1, 1, 1, 1] + [2, 2, 2, 2, 2, 2, 2],
            "round": [1, 2, 3, 4, 5, 6, 7] + [1, 2, 3, 4, 5, 6, 7],
            "total_points": [9, None, 3, 9, None, 6, 5] + [4, None, None, 3, 6, 9, 9],
        }
    )
    expected = players.with_columns(
        pl.Series(
            "total_points_rolling_mean_1",
            [0, 9, 9, 3, 9, 9, 6] + [0, 4, 4, 4, 3, 6, 9]
        )
    )
    players = players.sample(fraction=1.0, shuffle=True, seed=42)
    result = compute_rolling_mean(
        players,
        order_by="round",
        group_by="id",
        columns=["total_points"],
        window_sizes=[1],
        defaults=[0],
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
            "id": [1, 1, 1, 1, 1, 1, 1],
            "round": [1, 2, 3, 4, 5, 6, 7],
            "total_points": [9, 3, 3, None, None, None, None],
        }
    )
    expected = players.with_columns(
        pl.Series(
            "total_points_rolling_mean_3",
            [0, 9, 6, 5, 5, 5, 5],
        )
    )
    players = players.sample(fraction=1.0, shuffle=True, seed=42)
    result = compute_rolling_mean(
        players,
        order_by="round",
        group_by="id",
        columns=["total_points"],
        window_sizes=[3],
        defaults=[0],
    )
    assert_frame_equal(
        result,
        expected,
        check_row_order=False,
        check_column_order=False,
        check_exact=False,
        check_dtypes=False,
    )


def test_compute_availability():
    df = pl.DataFrame(
        {
            "season": ["2019-20", "2019-20", "2019-20"],
            "round": [1, 2, 3],
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
    _test_compute_availability(df, expected)

    df = df.with_columns(
        [
            pl.Series("chance_of_playing_next_round", [100, None, None]),
            pl.Series("status", ["a", None, None]),
            pl.Series("news", ["", None, None]),
            pl.Series("news_added", [datetime(2019, 8, 1), None, None]),
        ]
    )
    expected = pl.DataFrame({"availability": [100, 100, 100]})
    _test_compute_availability(df, expected)

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
    _test_compute_availability(df, expected)

    df = df.with_columns(
        [
            pl.Series("status", ["i", None, None]),
            pl.Series("news", ["Ankle injury - Expected back 15 Feb", None, None]),
            pl.Series("news_added", [datetime(2019, 8, 1), None, None]),
            pl.Series("chance_of_playing_next_round", [None, None, None]),
        ]
    )
    expected = pl.DataFrame({"availability": [0, 0, 0]})
    _test_compute_availability(df, expected)

    df = df.with_columns(
        [
            pl.Series("status", ["i", None, None]),
            pl.Series("news", ["Knee injury - Unknown return date", None, None]),
            pl.Series("news_added", [datetime(2019, 8, 1), None, None]),
            pl.Series("chance_of_playing_next_round", [None, None, None]),
        ]
    )
    expected = pl.DataFrame({"availability": [0, 0, 0]})
    _test_compute_availability(df, expected)

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
    _test_compute_availability(df, expected)

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
    _test_compute_availability(df, expected)

    df = df.with_columns(
        [
            pl.Series("status", ["s", None, None]),
            pl.Series("news", ["Suspended until 20 Feb", None, None]),
            pl.Series("news_added", [datetime(2019, 8, 1), None, None]),
            pl.Series("chance_of_playing_next_round", [0, None, None]),
        ]
    )
    expected = pl.DataFrame({"availability": [0, 0, 0]})
    _test_compute_availability(df, expected)

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
    _test_compute_availability(df, expected)

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
    _test_compute_availability(df, expected)

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
    _test_compute_availability(df, expected)

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
    _test_compute_availability(df, expected)

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
    _test_compute_availability(df, expected)

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
    _test_compute_availability(df, expected)

    # Test with multiple players
    df = pl.DataFrame(
        {
            "season": ["2019-20", "2019-20", "2019-20", "2019-20"],
            "round": [1, 1, 2, 2],
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
    _test_compute_availability(df, expected)

    # Test with null availability
    df = df.with_columns(
        [
            pl.Series("chance_of_playing_next_round", [None, None, None, 100]),
            pl.Series("status", [None, None, None, "a"]),
            pl.Series("news", [None, None, None, ""]),
        ]
    )
    expected = pl.DataFrame({"availability": [None, None, None, 100]})
    _test_compute_availability(df, expected)


def _test_compute_availability(df: pl.DataFrame, expected: pl.DataFrame):
    df = df.with_columns(pl.col("news_added").cast(pl.Datetime))
    result = compute_availability(df)
    result = result.select(expected.columns)
    assert_frame_equal(result, expected, check_dtypes=False)
