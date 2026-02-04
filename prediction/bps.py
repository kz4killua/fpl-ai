import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, RegressorMixin

from game.rules import DEF, FWD, GKP, MID
from prediction.total_points import apply_scoring_rules_to_predictions

BPS_RULES = {
    2024: {
        "1_to_59_minutes": {
            GKP: 3,
            DEF: 3,
            MID: 3,
            FWD: 3,
        },
        "60_plus_minutes": {
            GKP: 6,
            DEF: 6,
            MID: 6,
            FWD: 6,
        },
        "goals_scored": {
            GKP: 12,
            DEF: 12,
            MID: 18,
            FWD: 24,
        },
        "assists": {
            GKP: 9,
            DEF: 9,
            MID: 9,
            FWD: 9,
        },
        "clean_sheets": {
            GKP: 12,
            DEF: 12,
            MID: 0,
            FWD: 0,
        },
        "saves": {
            GKP: 2,
            DEF: 0,
            MID: 0,
            FWD: 0,
        },
        "goals_conceded": {
            GKP: -4,
            DEF: -4,
            MID: 0,
            FWD: 0,
        },
        "clearances_blocks_interceptions": {
            GKP: 0,
            DEF: 0,
            MID: 0,
            FWD: 0,
        },
        "tackles": {
            GKP: 0,
            DEF: 0,
            MID: 0,
            FWD: 0,
        },
        "recoveries": {
            GKP: 0,
            DEF: 0,
            MID: 0,
            FWD: 0,
        },
    },
    2025: {
        "1_to_59_minutes": {
            GKP: 3,
            DEF: 3,
            MID: 3,
            FWD: 3,
        },
        "60_plus_minutes": {
            GKP: 6,
            DEF: 6,
            MID: 6,
            FWD: 6,
        },
        "goals_scored": {
            GKP: 12,
            DEF: 12,
            MID: 18,
            FWD: 24,
        },
        "assists": {
            GKP: 9,
            DEF: 9,
            MID: 9,
            FWD: 9,
        },
        "clean_sheets": {
            GKP: 12,
            DEF: 12,
            MID: 0,
            FWD: 0,
        },
        "saves": {
            GKP: 2,
            DEF: 0,
            MID: 0,
            FWD: 0,
        },
        "goals_conceded": {
            GKP: -4,
            DEF: -4,
            MID: 0,
            FWD: 0,
        },
        "clearances_blocks_interceptions": {
            GKP: 1 / 2,
            DEF: 1 / 2,
            MID: 1 / 2,
            FWD: 1 / 2,
        },
        "tackles": {
            GKP: 2,
            DEF: 2,
            MID: 2,
            FWD: 2,
        },
        "recoveries": {
            GKP: 1 / 3,
            DEF: 1 / 3,
            MID: 1 / 3,
            FWD: 1 / 3,
        },
    },
}

# Fill in BPS rules for past seasons
for season in range(2016, 2024):
    BPS_RULES[season] = BPS_RULES[2024]


def make_bps_predictor():
    return BPSPredictor()


class BPSPredictor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        return apply_scoring_rules_to_predictions(X, BPS_RULES)
