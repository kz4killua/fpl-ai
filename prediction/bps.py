from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

from prediction.utils import RoutingEstimator, feature_selector


def make_bps_predictor():
    columns = [
        "element_type",
        "bps_rolling_mean_5",
        "bps_rolling_mean_20",
        "bps_mean_last_season",
        "predicted_minutes",
        "predicted_team_scored",
        "predicted_opponent_scored",
        "predicted_goals_scored",
        "predicted_assists",
        "predicted_saves",
    ]
    pipeline = make_pipeline(
        feature_selector(columns=columns),
        SimpleImputer(strategy="mean"),
        Ridge(random_state=42),
    )
    return RoutingEstimator(pipeline, "element_type")
