from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

from prediction.utils import RoutingEstimator, feature_selector


def make_total_points_predictor():
    columns = [
        "predicted_probability_1_to_60_minutes",
        "predicted_probability_60_plus_minutes",
        "predicted_player_clean_sheets",
        "predicted_player_goals_conceded",
        "predicted_goals_scored",
        "predicted_assists",
        "predicted_saves",
        "predicted_bonus",
    ]
    pipeline = make_pipeline(
        feature_selector(columns=columns),
        Ridge(alpha=1.0, random_state=42),
    )
    return RoutingEstimator(pipeline, "element_type")
