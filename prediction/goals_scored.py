from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from prediction.utils import FeatureSelector


def make_goals_scored_predictor():
    columns = [
        # General attributes
        "value",
        "was_home",
        # Intermediate predictions
        "predicted_team_scored",
        "predicted_1_to_59_minutes",
        "predicted_60_plus_minutes",
        # Balanced rolling means
        "balanced_goals_scored_rolling_mean_3",
        "balanced_goals_scored_rolling_mean_5",
        "balanced_goals_scored_rolling_mean_10",
        "balanced_goals_scored_rolling_mean_20",
        "balanced_uds_xG_rolling_mean_3",
        "balanced_uds_xG_rolling_mean_5",
        "balanced_uds_xG_rolling_mean_10",
        "balanced_uds_xG_rolling_mean_20",
        "balanced_threat_rolling_mean_3",
        "balanced_threat_rolling_mean_5",
        "balanced_threat_rolling_mean_10",
        "balanced_threat_rolling_mean_20",
        # Balanced rolling means of per-90 metrics
        "balanced_threat_per_90_rolling_mean_3",
        "balanced_threat_per_90_rolling_mean_5",
        "balanced_threat_per_90_rolling_mean_10",
        "balanced_threat_per_90_rolling_mean_20",
        # Defensive strengths of previous opponents
        "opponent_team_strength_defence_condition_rolling_mean_3",
        "opponent_team_strength_defence_condition_rolling_mean_5",
        "opponent_team_strength_defence_condition_rolling_mean_10",
        "opponent_team_strength_defence_condition_rolling_mean_20",
        # Set piece orders
        "imputed_penalties_order",
        "penalties_order_missing",
        "imputed_direct_freekicks_order",
        "direct_freekicks_order_missing",
        "imputed_corners_and_indirect_freekicks_order",
        "corners_and_indirect_freekicks_order_missing",
    ]
    model = Ridge(
        random_state=42,
    )
    return Pipeline(
        [
            ("features", FeatureSelector(columns)),
            ("impute", SimpleImputer(strategy="mean")),
            (
                "polynomial",
                PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
            ),
            ("scaler", StandardScaler()),
            ("predictor", model),
        ]
    )
