from lightgbm import LGBMRegressor
from sklearn.pipeline import make_pipeline

from prediction.utils import feature_selector


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
        )
    )
