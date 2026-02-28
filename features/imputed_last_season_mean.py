import polars as pl


def compute_imputed_last_season_mean(df: pl.LazyFrame, column: str) -> pl.LazyFrame:
    """Use a linear fit to approximate missing last season means."""

    # Compute starting values for each player in each season
    df = df.with_columns(
        pl.col("value")
        .sort_by("kickoff_time")
        .first()
        .over(["season", "element"])
        .alias("starting_value")
    )

    # Only fit on gameweek 1 to avoid leaking information from later gameweeks
    fit_data = df.filter((pl.col("gameweek") == 1) & pl.col(column).is_not_null())

    # Compute the fit coefficients (m and b) for each season and element type
    coefficients = (
        fit_data.group_by(["season", "element_type"])
        .agg(
            pl.len().alias("count"),
            pl.mean("starting_value").alias("x_mean"),
            pl.mean(column).alias("y_mean"),
            pl.var("starting_value").alias("var"),
            pl.cov("starting_value", column).alias("cov"),
        )
        .filter(pl.col("count") >= 2)
        .with_columns(
            pl.when(pl.col("var") == 0.0)
            .then(0.0)
            .otherwise(pl.col("cov") / pl.col("var"))
            .alias("m")
        )
        .with_columns((pl.col("y_mean") - pl.col("m") * pl.col("x_mean")).alias("b"))
        .select("season", "element_type", "m", "b")
    )
    df = df.join(coefficients, on=["season", "element_type"], how="left")

    # Compute the linear fit for missing entries
    df = df.with_columns(
        pl.when(pl.col(column).is_null())
        .then(pl.col("m") * pl.col("starting_value") + pl.col("b"))
        .otherwise(pl.col(column))
        .alias(f"imputed_{column}")
    )
    df = df.drop(["starting_value", "m", "b"])

    return df
