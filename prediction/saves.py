from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline

from prediction.utils import FeatureSelector, NonNegativeRegressor


def make_saves_predictor():
    columns = [
        "value",
        "record_count",
        "element_type",
        "predicted_1_to_59_minutes",
        "predicted_60_plus_minutes",
        "predicted_team_goals_scored",
        "predicted_opponent_goals_scored",
        "saves_rolling_mean_5",
        "saves_rolling_mean_20",
        "saves_mean_last_season",
    ]
    categorical_columns = [
        "element_type",
    ]
    categorical_features = [column in categorical_columns for column in columns]
    model = NonNegativeRegressor(
        HistGradientBoostingRegressor(
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.2,
            l2_regularization=0.1,
            max_leaf_nodes=7,
            min_samples_leaf=64,
            categorical_features=categorical_features,
            random_state=42,
        )
    )
    return make_pipeline(
        FeatureSelector(columns),
        model,
    )
