import numpy as np
import polars as pl
from sklearn.base import BaseEstimator
from sklearn.impute import KNNImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from prediction.utils import FeatureSelector


def make_team_goals_scored_predictor():
    return TeamGoalsScoredPredictor(
        base_model=make_base_team_goals_scored_predictor(), base_weight=0.2
    )


class TeamGoalsScoredPredictor(BaseEstimator):
    def __init__(self, base_model=None, base_weight=None):
        self.base_model = base_model
        self.base_weight = base_weight

    def get_meta_mask(self, X: pl.DataFrame) -> np.ndarray:
        return X["team_h_toa_expected_goals"].is_not_null().to_numpy()

    def fit(self, X: pl.DataFrame, y: pl.DataFrame):
        return self.base_model.fit(X, y)

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        # Predict goals scored using the base model
        base_preds = self.base_model.predict(X)

        # Get predicted / expected goals from market odds data
        meta_mask = self.get_meta_mask(X)
        if meta_mask.sum() == 0:
            return base_preds

        X_meta = X.filter(meta_mask)
        team_h_market_preds = X_meta["team_h_toa_expected_goals"].to_numpy()
        team_a_market_preds = X_meta["team_a_toa_expected_goals"].to_numpy()
        team_h_base_preds = base_preds[meta_mask, 0]
        team_a_base_preds = base_preds[meta_mask, 1]

        # Compute a weighted average of market and base model predictions
        meta_preds = np.zeros_like(base_preds[meta_mask, :])
        meta_preds[:, 0] = (
            self.base_weight * team_h_base_preds
            + (1 - self.base_weight) * team_h_market_preds
        )
        meta_preds[:, 1] = (
            self.base_weight * team_a_base_preds
            + (1 - self.base_weight) * team_a_market_preds
        )

        # Combine base and meta model predictions
        final_preds = base_preds.copy()
        final_preds[meta_mask, :] = meta_preds

        return final_preds


def make_base_team_goals_scored_predictor():
    columns = [
        # Club Elo features
        "team_h_clb_elo",
        "team_a_clb_elo",
        "team_h_clb_win_prob",
        "team_a_clb_win_prob",
        "team_h_clb_expected_goals",
        "team_a_clb_expected_goals",
        # XG
        "team_h_relative_uds_xG_rolling_mean_5",
        "team_a_relative_uds_xG_rolling_mean_5",
        "team_h_relative_uds_xG_rolling_mean_10",
        "team_a_relative_uds_xG_rolling_mean_10",
        "team_h_relative_uds_xG_rolling_mean_20",
        "team_a_relative_uds_xG_rolling_mean_20",
        "team_h_relative_uds_xG_rolling_mean_40",
        "team_a_relative_uds_xG_rolling_mean_40",
    ]
    model = MultiOutputRegressor(
        PoissonRegressor(alpha=1.0),
    )
    return Pipeline(
        [
            ("features", FeatureSelector(columns)),
            ("imputer", KNNImputer(n_neighbors=5)),
            ("scaler", StandardScaler()),
            ("predictor", model),
        ]
    )
