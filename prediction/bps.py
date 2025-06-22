from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline

from prediction.utils import feature_selector


def make_bps_predictor():
    columns = [
        "element_type",
        "predicted_minutes",
        "predicted_team_scored",
        "predicted_opponent_scored",
        "predicted_goals_scored",
        "predicted_assists",
        "predicted_saves"
    ]
    categorical_columns = [
        "element_type",
    ]
    categorical_features = [column in categorical_columns for column in columns]
    return make_pipeline(
        feature_selector(columns=columns),
        HistGradientBoostingRegressor(
            categorical_features=categorical_features,
            random_state=42,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.2,
            max_leaf_nodes=7,
            min_samples_leaf=1024,
            l2_regularization=10.0,
        ),
    )
