import polars as pl

from datautil.load.fpl import load_fpl


def test_load_fpl():
    # Load data for testing
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
    ).sort(["season", "element", "fixture"])
    result = (
        players.join(
            expected.select(["season", "element", "fixture"]),
            on=["season", "element", "fixture"],
            how="inner",
        )
        .select(expected.columns)
        .sort(["season", "element", "fixture"])
    )
    assert result.equals(expected)

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
