from datetime import datetime

import polars as pl
from polars.testing import assert_frame_equal

from features.availability import compute_availability
from features.per_90 import compute_per_90
from features.previous_season_mean import compute_previous_season_mean
from features.record_count import compute_record_count
from features.rolling_mean import compute_rolling_mean


def test_compute_record_count():
    # Test with a single player (no null values)
    df = pl.DataFrame(
        {
            "season": ["2021-22"] * 5,
            "kickoff_time": [1, 2, 3, 4, 5],
            "code": [1, 1, 1, 1, 1],
            "total_points": [0, 0, 0, 0, 0],
        }
    )
    expected = df.with_columns(pl.Series("record_count", [0, 1, 2, 3, 4]))
    result = compute_record_count(df)
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
            "season": ["2021-22"] * 5,
            "kickoff_time": [1, 2, 3, 4, 5],
            "code": [1, 1, 1, 1, 1],
            "total_points": [0, 0, 0, None, None],
        }
    )
    expected = df.with_columns(pl.Series("record_count", [0, 1, 2, 3, 3]))
    result = compute_record_count(df)
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
            "season": ["2021-22"] * 3 + ["2022-23"] * 3,
            "kickoff_time": [1, 2, 3, 4, 5, 6],
            "code": [1, 1, 1, 1, 1, 1],
            "total_points": [2, 2, 2, 2, 2, 2],
        }
    )
    expected = df.with_columns(pl.Series("record_count", [0, 1, 2, 0, 1, 2]))
    result = compute_record_count(df)
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
            "season": ["2021-22"] * 6,
            "kickoff_time": [1, 1, 2, 2, 3, 3],
            "code": [1, 2, 1, 2, 1, 2],
            "total_points": [2, 2, 2, 2, None, None],
        }
    )
    expected = df.with_columns(pl.Series("record_count", [0, 0, 1, 1, 2, 2]))
    result = compute_record_count(df)
    assert_frame_equal(
        result,
        expected,
        check_row_order=False,
        check_column_order=False,
        check_dtypes=False,
    )


def test_compute_rolling_mean():
    # Test rolling means with multiple players
    players = pl.DataFrame(
        {
            "season": ["2021-22"] * 14,
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
            "season": ["2021-22"] * 14,
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
            "season": ["2021-22"] * 7,
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
            "season": ["2021-22"] * 3 + ["2022-23"] * 3,
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


def test_compute_previous_season_mean():
    # Test with a single player
    players = pl.DataFrame(
        {
            "code": [1, 1, 1, 1],
            "round": [1, 2, 1, 2],
            "season": ["2021-22", "2021-22", "2022-23", "2022-23"],
            "total_points": [9, 1, None, None],
        }
    )
    expected = players.with_columns(
        pl.Series(
            "previous_season_mean_total_points",
            [None, None, 5, 5],
        )
    )
    players = players.sample(fraction=1.0, shuffle=True, seed=42)
    result = compute_previous_season_mean(players, ["total_points"])
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
            "round": [1, 1, 1, 1],
            "season": ["2021-22", "2021-22", "2022-23", "2022-23"],
            "total_points": [2, 5, 3, 7],
        }
    )
    expected = players.with_columns(
        pl.Series(
            "previous_season_mean_total_points",
            [None, None, 2, 5],
        )
    )
    players = players.sample(fraction=1.0, shuffle=True, seed=42)
    result = compute_previous_season_mean(players, ["total_points"])
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
            "id": [1, 2, 3, 4, 5],
            "minutes": [0, 90, 90, 45, 90],
            "goals": [0, 0, 2, 2, 2],
        }
    )
    expected = pl.DataFrame(
        {
            "goals_per_90": [0, 0, 2, 4, 2],
        }
    )
    result = compute_per_90(players, ["goals"])
    result = result.select(expected.columns)
    assert_frame_equal(result, expected, check_dtypes=False)


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
