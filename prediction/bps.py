from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

from prediction.utils import FeatureSelector, RoutingEstimator


def make_bps_predictor():
    columns = [
        "element_type",
        "bps_rolling_mean_5",
        "bps_rolling_mean_20",
        "bps_mean_last_season",
        "predicted_1_to_59_minutes",
        "predicted_60_plus_minutes",
        "predicted_team_scored",
        "predicted_opponent_scored",
        "predicted_goals_scored",
        "predicted_assists",
        "predicted_saves",
    ]
    pipeline = make_pipeline(
        FeatureSelector(columns),
        SimpleImputer(strategy="mean"),
        Ridge(random_state=42),
    )
    return RoutingEstimator(pipeline, "element_type")
