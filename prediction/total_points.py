from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

from prediction.utils import FeatureSelector, RoutingEstimator


def make_total_points_predictor():
    columns = [
        "predicted_1_to_59_minutes",
        "predicted_60_plus_minutes",
        "predicted_clean_sheets",
        "predicted_goals_conceded",
        "predicted_goals_scored",
        "predicted_assists",
        "predicted_saves",
        "predicted_bonus",
    ]
    pipeline = make_pipeline(
        FeatureSelector(columns),
        Ridge(alpha=1.0, random_state=42),
    )
    return RoutingEstimator(pipeline, "element_type")
