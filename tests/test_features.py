import polars as pl
from polars.testing import assert_frame_equal

from features.rolling_mean import rolling_mean


def test_rolling_mean():
    # Prepare a dataset of two players and seven gameweeks
    players = pl.DataFrame(
        {
            "id": [1, 1, 1, 1, 1, 1, 1] + [2, 2, 2, 2, 2, 2, 2],
            "round": [1, 2, 3, 4, 5, 6, 7] + [1, 2, 3, 4, 5, 6, 7],
            "total_points": [9, 5, 3, 9, 8, 6, 5] + [4, 6, 2, 3, 6, 9, 9],
        }
    )
    # Rolling means must be unaffected by the order of the rows
    players = players.sample(fraction=1.0, shuffle=True, seed=42)

    # Test values
    expected = pl.DataFrame(
        {
            "id": [1, 1, 1, 1, 1, 1, 1] + [2, 2, 2, 2, 2, 2, 2],
            "round": [1, 2, 3, 4, 5, 6, 7] + [1, 2, 3, 4, 5, 6, 7],
            "total_points": [9, 5, 3, 9, 8, 6, 5] + [4, 6, 2, 3, 6, 9, 9],
            "total_points_rolling_mean_3": [
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
            ],
            "total_points_rolling_mean_5": [
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
            ],
        }
    )
    result = rolling_mean(
        players,
        order_by="round",
        group_by="id",
        columns=["total_points", "total_points"],
        windows=[3, 5],
        defaults=[0, 0],
    )
    assert_frame_equal(
        result,
        expected,
        check_row_order=False,
        check_column_order=False,
        check_exact=False,
    )
