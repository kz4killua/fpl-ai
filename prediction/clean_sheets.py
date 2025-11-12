import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, ClassifierMixin


class CleanSheetsPredictor(BaseEstimator, ClassifierMixin):
    def fit(self, X, y=None):
        return self

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        return (
            X.get_column("predicted_team_clean_sheets")
            * X.get_column("predicted_60_plus_minutes")
        ).to_numpy()


def make_clean_sheets_predictor():
    return CleanSheetsPredictor()
