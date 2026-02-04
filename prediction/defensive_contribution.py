import numpy as np
import polars as pl
from scipy.stats import poisson
from sklearn.base import BaseEstimator, RegressorMixin

from game.rules import DEF, FWD, GKP, MID

THRESHOLDS = {
    DEF: 10,
    MID: 12,
    FWD: 12,
}


class DefensiveContributionPredictor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        y = np.zeros(len(X), dtype=np.float64)

        gkp_mask = (X["element_type"] == GKP).to_numpy()
        def_mask = (X["element_type"] == DEF).to_numpy()
        mid_mask = (X["element_type"] == MID).to_numpy()
        fwd_mask = (X["element_type"] == FWD).to_numpy()

        cbi = X["predicted_clearances_blocks_interceptions"].to_numpy()
        t = X["predicted_tackles"].to_numpy()
        r = X["predicted_recoveries"].to_numpy()

        y[gkp_mask] = 0.0
        y[def_mask] = cbi[def_mask] + t[def_mask]
        y[mid_mask] = cbi[mid_mask] + t[mid_mask] + r[mid_mask]
        y[fwd_mask] = cbi[fwd_mask] + t[fwd_mask] + r[fwd_mask]

        return y


def make_defensive_contribution_predictor():
    return DefensiveContributionPredictor()


def calculate_defensive_contribution_threshold_prob(
    df: pl.DataFrame,
    defensive_contribution_column: str,
):
    probs = np.zeros(len(df), dtype=np.float64)

    masks = {
        GKP: (df["element_type"] == GKP).to_numpy(),
        DEF: (df["element_type"] == DEF).to_numpy(),
        MID: (df["element_type"] == MID).to_numpy(),
        FWD: (df["element_type"] == FWD).to_numpy(),
    }

    # Apply thresholds based on the player's position
    defensive_contribution = df[defensive_contribution_column].to_numpy()
    for position in THRESHOLDS:
        probs[masks[position]] = poisson.sf(
            THRESHOLDS[position] - 1, defensive_contribution[masks[position]]
        )

    return probs
