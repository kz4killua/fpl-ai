from lightgbm import LGBMRegressor
from sklearn.pipeline import make_pipeline

from prediction.utils import feature_selector


def make_goals_scored_predictor():
    columns = [
        "value",
        "record_count",
        "predicted_1_to_59_minutes",
        "predicted_60_plus_minutes",
        "predicted_team_scored",
        "uds_xG_rolling_mean_5",
        "uds_xG_rolling_mean_20",
        "uds_xG_mean_last_season",
    ]
    return make_pipeline(
        feature_selector(columns=columns),
        LGBMRegressor(
            random_state=42,
            min_child_samples=512,
            num_leaves=7,
            reg_alpha=10,
            reg_lambda=0,
            verbosity=-1,
        ),
    )
