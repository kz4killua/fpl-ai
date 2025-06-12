from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from prediction.utils import ElementTypeSplitEstimator, feature_selector


def make_total_points_predictor():
    pipeline = make_pipeline(
        feature_selector(
            [
                # Intermediate predictions
                "predicted_minutes",
                "predicted_team_scored",
                "predicted_opponent_scored",
                "predicted_team_clean_sheets",
                "predicted_opponent_clean_sheets",
                # Player features
                "minutes_rolling_mean_3",
                "total_points_rolling_mean_5",
                "uds_xG_rolling_mean_5",
                "uds_xA_rolling_mean_5",
                "clean_sheets_rolling_mean_5",
                "goals_conceded_rolling_mean_5",
                "saves_rolling_mean_5",
                "bonus_rolling_mean_5",
                "influence_rolling_mean_5",
                "creativity_rolling_mean_5",
                "threat_rolling_mean_5",
                "ict_index_rolling_mean_5",
            ]
        ),
        StandardScaler(),
        LinearRegression(),
    )
    return ElementTypeSplitEstimator(pipeline)
