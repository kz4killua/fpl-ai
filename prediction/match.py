from sklearn.impute import KNNImputer
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from prediction.utils import FeatureSelector


def make_match_predictor():
    columns = [
        "team_h_relative_attack_strength",
        "team_a_relative_attack_strength",
        "team_h_relative_uds_xG_rolling_mean_10",
        "team_a_relative_uds_xG_rolling_mean_10",
        "team_h_relative_uds_xG_rolling_mean_30",
        "team_a_relative_uds_xG_rolling_mean_30",
        "team_h_relative_scored_rolling_mean_10",
        "team_a_relative_scored_rolling_mean_10",
        "team_h_relative_scored_rolling_mean_30",
        "team_a_relative_scored_rolling_mean_30",
    ]
    return make_pipeline(
        FeatureSelector(columns),
        KNNImputer(n_neighbors=5),
        StandardScaler(),
        MultiOutputRegressor(
            Ridge(alpha=100),
        ),
    )
