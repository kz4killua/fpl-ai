from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from prediction.utils import FeatureSelector


def make_recoveries_predictor():
    columns = [
        # Intermediate predictions
        "predicted_1_to_59_minutes",
        "predicted_60_plus_minutes",
        "predicted_opponent_goals_scored",
        # Balanced rolling means of per 90 metrics
        "balanced_recoveries_per_90_rolling_mean_3",
        "balanced_recoveries_per_90_rolling_mean_5",
        "balanced_recoveries_per_90_rolling_mean_10",
        "balanced_recoveries_per_90_rolling_mean_20",
    ]
    model = PoissonRegressor(alpha=1e-3, max_iter=1000)
    return Pipeline(
        [
            ("features", FeatureSelector(columns)),
            (
                "polynomial",
                PolynomialFeatures(degree=2, include_bias=False, interaction_only=True),
            ),
            ("scaler", StandardScaler()),
            ("predictor", model),
        ]
    )
