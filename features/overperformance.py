import polars as pl


def _compute_overperformance(
    df,
    expected_col_name: str,
    actual_col_name: str,
    stability: int = 20,
):
    """Computes a ratio of actual to expected values for each player, adjusted for small
    sample sizes."""
    df = df.sort(["code", "kickoff_time"])

    # Compute the long term ratio of actual (e.g. goals scored) to expected values
    # (e.g. xG) for each player
    df = df.with_columns(
        pl.col(actual_col_name)
        .fill_null(0)
        .cum_sum()
        .shift(1, fill_value=0)
        .over("code")
        .alias("_cum_sum_actual"),
        pl.col(expected_col_name)
        .fill_null(0)
        .cum_sum()
        .shift(1, fill_value=0)
        .over("code")
        .alias("_cum_sum_expected"),
    )

    df = df.with_columns(
        pl.when(pl.col("_cum_sum_expected") > 0.01)
        .then(pl.col("_cum_sum_actual") / pl.col("_cum_sum_expected"))
        .otherwise(None)
        .alias("_raw_overperformance")
    )

    # Compute alpha, the measure of how much we trust the long term ratio
    df = df.with_columns(
        (pl.col("_cum_sum_expected") / (pl.col("_cum_sum_expected") + stability)).alias(
            "_alpha"
        )
    )

    # Combine the long term ratio with the prior of 1.0, weighted by alpha
    df = df.with_columns(
        pl.when(pl.col("_raw_overperformance").is_not_null())
        .then(
            pl.col("_alpha") * pl.col("_raw_overperformance")
            + (1 - pl.col("_alpha")) * 1.0
        )
        .otherwise(1.0)
        .alias(f"{expected_col_name}_overperformance")
    )

    df = df.drop(
        "_cum_sum_actual", "_cum_sum_expected", "_raw_overperformance", "_alpha"
    )

    return df


def compute_uds_xg_overperformance(df):
    return _compute_overperformance(df, "uds_xG", "goals_scored", stability=20)


def compute_uds_xa_overperformance(df):
    return _compute_overperformance(df, "uds_xA", "assists", stability=20)


def compute_adjusted_uds_xg(df):
    """Adjust the xG for each player based on how their long term overperformance."""
    df = compute_uds_xg_overperformance(df)
    df = df.with_columns(
        (pl.col("uds_xG") * pl.col("uds_xG_overperformance")).alias("adjusted_uds_xG")
    )
    return df


def compute_adjusted_uds_xa(df):
    """Adjust the xA for each player based on how their long term overperformance."""
    df = compute_uds_xa_overperformance(df)
    df = df.with_columns(
        (pl.col("uds_xA") * pl.col("uds_xA_overperformance")).alias("adjusted_uds_xA")
    )
    return df
