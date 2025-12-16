from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from prediction.utils import FeatureSelector


def make_assists_predictor():
    columns = [
        # Intermediate predictions
        "predicted_team_goals_scored",
        "predicted_1_to_59_minutes",
        "predicted_60_plus_minutes",
        # Balanced rolling means of per-90 metrics
        "balanced_assists_per_90_rolling_mean_3",
        "balanced_assists_per_90_rolling_mean_5",
        "balanced_assists_per_90_rolling_mean_10",
        "balanced_assists_per_90_rolling_mean_20",
        "balanced_uds_xA_per_90_rolling_mean_3",
        "balanced_uds_xA_per_90_rolling_mean_5",
        "balanced_uds_xA_per_90_rolling_mean_10",
        "balanced_uds_xA_per_90_rolling_mean_20",
        "balanced_creativity_per_90_rolling_mean_3",
        "balanced_creativity_per_90_rolling_mean_5",
        "balanced_creativity_per_90_rolling_mean_10",
        "balanced_creativity_per_90_rolling_mean_20",
        # Balanced rolling means of share metrics
        "balanced_assists_share_rolling_mean_3",
        "balanced_assists_share_rolling_mean_5",
        "balanced_assists_share_rolling_mean_10",
        "balanced_assists_share_rolling_mean_20",
        "balanced_uds_xA_share_rolling_mean_3",
        "balanced_uds_xA_share_rolling_mean_5",
        "balanced_uds_xA_share_rolling_mean_10",
        "balanced_uds_xA_share_rolling_mean_20",
        "balanced_creativity_share_rolling_mean_3",
        "balanced_creativity_share_rolling_mean_5",
        "balanced_creativity_share_rolling_mean_10",
        "balanced_creativity_share_rolling_mean_20",
        # Set piece orders
        "imputed_penalties_order",
        "penalties_order_missing",
        "imputed_direct_freekicks_order",
        "direct_freekicks_order_missing",
        "imputed_corners_and_indirect_freekicks_order",
        "corners_and_indirect_freekicks_order_missing",
    ]
    model = Ridge(
        alpha=10000,
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
