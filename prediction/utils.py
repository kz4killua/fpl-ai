import pickle
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import BaseCrossValidator

MODELS_DIR = Path("models")


def save_model(model: BaseEstimator, name: str):
    """Save the model to a file."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(name: str) -> BaseEstimator:
    """Load the model from a file."""
    path = MODELS_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model {name} not found at {path}.")
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


def feature_selector(columns: list[str]) -> ColumnTransformer:
    """Select features for the model."""
    return ColumnTransformer(
        transformers=[
            ("features", "passthrough", columns),
        ],
        remainder="drop",
        sparse_threshold=0,
    )


def get_season_splits(seasons: list[str]):
    """Get the training and testing seasons for cross-validation."""
    seasons = sorted(set(seasons))
    splits = [(seasons[:i], seasons[i]) for i in range(1, len(seasons))]
    yield from splits


class SeasonSplit(BaseCrossValidator):
    """Like `TimeSeriesSplit`, but splits by seasons"""

    def __init__(self, seasons: list[str]):
        self.seasons = sorted(set(seasons))

    def split(self, X: pl.DataFrame, y=None, groups=None):
        """Splits the data into training and testing sets based on seasons."""
        for train_seasons, test_season in get_season_splits(self.seasons):
            train_idx = X["season"].is_in(train_seasons).arg_true().to_numpy()
            test_idx = (X["season"] == test_season).arg_true().to_numpy()
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splits."""
        return len(self.seasons) - 1


class RoutingEstimator(BaseEstimator):
    """Uses a different estimator for each unique value in a column."""

    def __init__(self, base: BaseEstimator, column: str):
        self.base = base
        self.column = column
        self.estimators = {}

    def fit(self, X: pl.DataFrame, y: pl.Series):
        """Fit the base estimator for each unique value in the column."""
        # Get all unique values in the specified column
        unique = X.get_column(self.column).unique().to_list()
        if len(unique) > 10:
            raise ValueError(
                f"Too many unique values in column '{self.column}': {len(unique)}"
            )
        # Fit a separate estimator for each unique value
        for value in unique:
            mask = X[self.column] == value
            X_value = X.filter(mask)
            y_value = y.filter(mask)
            estimator = clone(self.base)
            estimator.fit(X_value, y_value)
            self.estimators[value] = estimator

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        """Predict using the appropriate estimator for each row."""
        y = np.zeros(len(X))
        for value, estimator in self.estimators.items():
            mask = X[self.column] == value
            if mask.any():
                X_value = X.filter(mask)
                y[mask.to_numpy()] = estimator.predict(X_value)
        return y


class ConditionalRegressor(BaseEstimator, RegressorMixin):
    """
    Returns the prediction of an estimator if a condition is met,
    otherwise returns a default value.
    """

    def __init__(self, estimator: BaseEstimator, condition: pl.Expr, default: float):
        self.estimator = estimator
        self.condition = condition
        self.default = default

    def fit(self, X: pl.DataFrame, y: pl.Series):
        """Fit the estimator only on the rows where the condition is met."""
        mask = X.select(self.condition).to_series()
        if not mask.any():
            raise ValueError("No rows match the condition for fitting.")
        print(f"Fitting estimator on {mask.sum()}/{len(mask)} rows.")

        X_condition = X.filter(mask)
        y_condition = y.filter(mask)
        self.estimator.fit(X_condition, y_condition)
        return self

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        """Predict using the estimator only where the condition is met."""
        mask = X.select(self.condition).to_series()
        print(f"Predicting on {mask.sum()}/{len(mask)} rows.")
        predictions = np.full(X.height, self.default, dtype=np.float64)
        X_condition = X.filter(mask)
        predictions[mask.to_numpy()] = self.estimator.predict(X_condition)
        return predictions
