import itertools

import numpy as np
import polars as pl


def compute_imputed_last_season_mean(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """Use a linear fit to approximate missing last season means."""
    seasons = sorted(df.get_column("season").unique())
    seasons = seasons[1:]
    element_types = sorted(df.get_column("element_type").unique())

    # Compute the starting values for each player in each season
    starting_values = df.sort("kickoff_time").group_by(["season", "element"]).first()
    df = df.join(
        starting_values.select(
            [
                pl.col("season"),
                pl.col("element"),
                pl.col("value").alias("starting_value"),
            ]
        ),
        on=["season", "element"],
        how="left",
    )

    # Initialize a new column for imputed values
    alias = f"imputed_{column}"
    df = df.with_columns(pl.col(column).alias(alias))

    # Get linear fit coefficients for each season and element type
    for season, element_type in itertools.product(seasons, element_types):
        filters = (pl.col("season") == season) & (
            pl.col("element_type") == element_type
        )
        filtered = df.filter(
            filters
            & pl.col(column).is_not_null()
            &
            # Note: This is important to prevent data leakage.
            pl.col("round")
            == 1
        )

        # Create a linear fit between "starting_value" and the target column
        x = filtered.get_column("starting_value").to_numpy()
        y = filtered.get_column(column).to_numpy()
        m, b = np.polyfit(x, y, deg=1)

        # Use the fit coefficients to fill in missing values
        df = df.with_columns(
            pl.when(filters & pl.col(alias).is_null())
            .then(m * pl.col("starting_value") + b)
            .otherwise(pl.col(alias))
            .alias(alias)
        )

    # Drop temporary columns
    df = df.drop("starting_value")

    return df
