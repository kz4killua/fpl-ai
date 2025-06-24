from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

from prediction.utils import RoutingEstimator, feature_selector


def make_total_points_predictor():
    columns = [
        "element_type",
        "predicted_minutes",
        "predicted_team_scored",
        "predicted_opponent_scored",
        "predicted_team_clean_sheets",
        "predicted_opponent_clean_sheets",
        "predicted_goals_scored",
        "predicted_assists",
        "predicted_saves",
        "predicted_bps",
        "predicted_bps_rank",
    ]
    pipeline = make_pipeline(
        feature_selector(columns=columns),
        Ridge(
            alpha=10.0,
            random_state=42,
        ),
    )
    return RoutingEstimator(pipeline, "element_type")
