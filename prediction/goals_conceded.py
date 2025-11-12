import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, RegressorMixin


class GoalsConcededPredictor(BaseEstimator, RegressorMixin):
    def fit(self, X, y=None):
        return self

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        return (
            X.get_column("predicted_opponent_goals_scored")
            * X.get_column("predicted_60_plus_minutes")
        ).to_numpy()


def make_goals_conceded_predictor():
    return GoalsConcededPredictor()
