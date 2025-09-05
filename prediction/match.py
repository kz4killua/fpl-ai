from sklearn.impute import KNNImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from prediction.utils import FeatureSelector


def make_match_predictor():
    columns = [
        "home_market_probability",
        "away_market_probability",
        "draw_market_probability",
        "team_h_clb_elo",
        "team_a_clb_elo",
        "team_h_clb_elo_win_probability",
        "team_a_clb_elo_win_probability",
        "team_h_relative_uds_xG_rolling_mean_5",
        "team_a_relative_uds_xG_rolling_mean_5",
        "team_h_relative_uds_xG_rolling_mean_10",
        "team_a_relative_uds_xG_rolling_mean_10",
        "team_h_relative_uds_xG_rolling_mean_20",
        "team_a_relative_uds_xG_rolling_mean_20",
        "team_h_relative_uds_xG_rolling_mean_40",
        "team_a_relative_uds_xG_rolling_mean_40",
        "team_h_relative_scored_rolling_mean_5",
        "team_a_relative_scored_rolling_mean_5",
        "team_h_relative_scored_rolling_mean_10",
        "team_a_relative_scored_rolling_mean_10",
        "team_h_relative_scored_rolling_mean_20",
        "team_a_relative_scored_rolling_mean_20",
        "team_h_relative_scored_rolling_mean_40",
        "team_a_relative_scored_rolling_mean_40",
    ]
    model = MultiOutputRegressor(
        PoissonRegressor(alpha=1),
    )
    return Pipeline(
        [
            ("features", FeatureSelector(columns)),
            ("imputer", KNNImputer(n_neighbors=5)),
            ("scaler", StandardScaler()),
            ("predictor", model),
        ]
    )
