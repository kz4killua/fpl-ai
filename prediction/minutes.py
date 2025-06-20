from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline

from prediction.utils import feature_selector


def make_minutes_predictor():
    columns = [
        "element_type",
        "availability",
        "record_count",
        "previous_season_mean_minutes",
        "previous_season_mean_starts",
        "starts_rolling_mean_3",
        "starts_rolling_mean_10",
        "minutes_rolling_mean_3",
        "minutes_rolling_mean_10",
    ]
    categorical_columns = [
        "element_type",
    ]
    # Convert categorical features to a mask
    categorical_features = [column in categorical_columns for column in columns]
    return make_pipeline(
        feature_selector(columns=columns),
        HistGradientBoostingRegressor(
            categorical_features=categorical_features,
            random_state=42,
        ),
    )
