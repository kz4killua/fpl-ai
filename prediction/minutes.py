import polars as pl
from lightgbm import LGBMRegressor
from scipy.stats import norm
from sklearn.pipeline import make_pipeline

from prediction.utils import feature_selector

STD_DEV_MINUTES = 25.0


def make_minutes_predictor():
    columns = [
        "value",
        "element_type",
        "availability",
        "record_count",
        "minutes_rolling_mean_3",
        "minutes_rolling_mean_10",
        "minutes_mean_last_season",
        "starts_rolling_mean_3",
        "starts_rolling_mean_10",
        "starts_mean_last_season",
    ]
    return make_pipeline(
        feature_selector(columns=columns),
        LGBMRegressor(
            random_state=42,
            min_child_samples=32,
            num_leaves=15,
            reg_alpha=100.0,
            reg_lambda=100.0,
            verbosity=-1,
        ),
    )


def compute_predicted_minute_probabilities(df: pl.DataFrame):
    # Assume a normal distribution for the predicted minutes
    predicted_minutes = df["predicted_minutes"].to_numpy()
    dist = norm(loc=predicted_minutes, scale=STD_DEV_MINUTES)

    # Compute the probabilities for each minute range
    predicted_probability_0_minutes = dist.cdf(0)
    predicted_probability_1_to_60_minutes = dist.cdf(60) - dist.cdf(0)
    predicted_probability_60_plus_minutes = 1 - dist.cdf(60)

    # Return the probabilities for each minute range
    df = df.with_columns(
        pl.Series("predicted_probability_0_minutes", predicted_probability_0_minutes),
        pl.Series(
            "predicted_probability_1_to_60_minutes",
            predicted_probability_1_to_60_minutes,
        ),
        pl.Series(
            "predicted_probability_60_plus_minutes",
            predicted_probability_60_plus_minutes,
        ),
    )
    return df
