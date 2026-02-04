import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, RegressorMixin

from game.rules import DEF, FWD, GKP, MID

TOTAL_POINTS_RULES = {
    2024: {
        "1_to_59_minutes": {
            GKP: 1,
            DEF: 1,
            MID: 1,
            FWD: 1,
        },
        "60_plus_minutes": {
            GKP: 2,
            DEF: 2,
            MID: 2,
            FWD: 2,
        },
        "goals_scored": {
            GKP: 10,
            DEF: 6,
            MID: 5,
            FWD: 4,
        },
        "assists": {
            GKP: 3,
            DEF: 3,
            MID: 3,
            FWD: 3,
        },
        "clean_sheets": {
            GKP: 4,
            DEF: 4,
            MID: 1,
            FWD: 0,
        },
        "saves": {
            GKP: 1 / 3,
            DEF: 0,
            MID: 0,
            FWD: 0,
        },
        "goals_conceded": {
            GKP: -1 / 2,
            DEF: -1 / 2,
            MID: 0,
            FWD: 0,
        },
        "bonus": {
            GKP: 1,
            DEF: 1,
            MID: 1,
            FWD: 1,
        },
        "defensive_contribution_threshold_prob": {
            GKP: 0,
            DEF: 0,
            MID: 0,
            FWD: 0,
        },
    },
    2025: {
        "1_to_59_minutes": {
            GKP: 1,
            DEF: 1,
            MID: 1,
            FWD: 1,
        },
        "60_plus_minutes": {
            GKP: 2,
            DEF: 2,
            MID: 2,
            FWD: 2,
        },
        "goals_scored": {
            GKP: 10,
            DEF: 6,
            MID: 5,
            FWD: 4,
        },
        "assists": {
            GKP: 3,
            DEF: 3,
            MID: 3,
            FWD: 3,
        },
        "clean_sheets": {
            GKP: 4,
            DEF: 4,
            MID: 1,
            FWD: 0,
        },
        "saves": {
            GKP: 1 / 3,
            DEF: 0,
            MID: 0,
            FWD: 0,
        },
        "goals_conceded": {
            GKP: -1 / 2,
            DEF: -1 / 2,
            MID: 0,
            FWD: 0,
        },
        "bonus": {
            GKP: 1,
            DEF: 1,
            MID: 1,
            FWD: 1,
        },
        "defensive_contribution_threshold_prob": {
            GKP: 0,
            DEF: 2,
            MID: 2,
            FWD: 2,
        },
    },
}

# Fill in total points rules for past seasons
for season in range(2016, 2024):
    TOTAL_POINTS_RULES[season] = TOTAL_POINTS_RULES[2024]


class TotalPointsPredictor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        y = np.zeros(len(X), dtype=np.float64)

        # Get masks for each season and element type
        seasons = sorted(X["season"].unique().to_list())
        element_types = [GKP, DEF, MID, FWD]
        season_masks = {
            season: (X["season"] == season).to_numpy() for season in seasons
        }
        element_type_masks = {
            element_type: (X["element_type"] == element_type).to_numpy()
            for element_type in element_types
        }

        # Pull out all intermediate predictions
        actions = [
            "1_to_59_minutes",
            "60_plus_minutes",
            "goals_scored",
            "assists",
            "clean_sheets",
            "saves",
            "goals_conceded",
            "defensive_contribution_threshold_prob",
            "bonus",
        ]
        predicted_values = {
            action: X[f"predicted_{action}"].to_numpy() for action in actions
        }

        # Get the scoring rules for each season
        for season in seasons:
            season_rules = TOTAL_POINTS_RULES[season]

            # Award points for each action, depending on the element type
            for action in actions:
                for element_type, multiplier in season_rules[action].items():
                    mask = season_masks[season] & element_type_masks[element_type]
                    y[mask] += predicted_values[action][mask] * multiplier

        return y


def make_total_points_predictor():
    return TotalPointsPredictor()
