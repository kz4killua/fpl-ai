from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline

from prediction.utils import feature_selector


def make_minutes_predictor():
    return make_pipeline(
        feature_selector(
            [
                "element_type",
                "availability",
                "minutes_rolling_mean_3",
            ]
        ),
        HistGradientBoostingRegressor(random_state=42),
    )
