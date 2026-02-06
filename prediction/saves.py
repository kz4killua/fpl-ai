import polars as pl
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from game.rules import GKP
from prediction.utils import ConditionalRegressor, FeatureSelector, NonNegativeRegressor


def make_saves_predictor():
    columns = [
        "predicted_1_to_59_minutes",
        "predicted_60_plus_minutes",
        "predicted_team_goals_scored",
        "predicted_opponent_goals_scored",
        "balanced_saves_per_90_rolling_mean_3",
        "balanced_saves_per_90_rolling_mean_5",
        "balanced_saves_per_90_rolling_mean_10",
        "balanced_saves_per_90_rolling_mean_20",
    ]
    model = NonNegativeRegressor(Ridge(alpha=100, random_state=42))
    pipeline = Pipeline(
        [
            ("features", FeatureSelector(columns)),
            (
                "polynomial",
                PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
            ),
            ("scaler", StandardScaler()),
            ("predictor", model),
        ]
    )
    return ConditionalRegressor(pipeline, (pl.col("element_type") == GKP), 0.0)
