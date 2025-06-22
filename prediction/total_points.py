from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline

from prediction.utils import feature_selector


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
    categorical_columns = [
        "element_type",
    ]
    categorical_features = [column in categorical_columns for column in columns]
    pipeline = make_pipeline(
        feature_selector(columns=columns),
        HistGradientBoostingRegressor(
            categorical_features=categorical_features,
            random_state=42,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.2,
            l2_regularization=1.0,
            max_leaf_nodes=7,
            min_samples_leaf=256,
        )
    )
    return pipeline