import pickle
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
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


def get_season_splits(seasons: list[str]):
    """Get the training and testing seasons for cross-validation."""
    seasons = sorted(set(seasons))
    splits = [(seasons[:i], seasons[i]) for i in range(1, len(seasons))]
    yield from splits


class FeatureSelector(TransformerMixin):
    """Selects a list of columns from a Polars DataFrame."""

    def __init__(self, columns: list[str]):
        self.columns = columns
        self.transformer = ColumnTransformer(
            transformers=[
                ("features", "passthrough", self.columns),
            ],
            remainder="drop",
            sparse_threshold=0,
        )

    def fit(self, X: pl.DataFrame, y=None):
        """Fit the transformer (no-op for FeatureSelector)."""
        return self.transformer.fit(X)

    def transform(self, X: pl.DataFrame, y=None) -> pl.DataFrame:
        return self.transformer.transform(X)


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


class NonNegativeRegressor(BaseEstimator, RegressorMixin):
    """Clips the predictions of a regressor to be non-negative."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        return self

    def predict(self, X):
        predictions = self.estimator_.predict(X)
        return np.maximum(0, predictions)


class ConditionalRegressor(BaseEstimator, RegressorMixin):
    """
    Fits and predicts only on the rows where the condition is met.
    Returns a default value for all other rows.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        condition: pl.Expr,
        default: float = 0.0,
        verbose: bool = False,
    ):
        self.estimator = estimator
        self.condition = condition
        self.default = default
        self.verbose = verbose

    def fit(self, X: pl.DataFrame, y: pl.Series):
        """Fit the estimator only on the rows where the condition is met."""
        self.check_input_types(X, y)

        mask = X.select(self.condition).to_series()
        if not mask.any():
            raise ValueError("No rows match the condition for fitting.")

        if self.verbose:
            print(f"Fitting estimator on {mask.sum()}/{len(mask)} rows.")

        self.estimator_ = clone(self.estimator)

        X_condition = X.filter(mask)
        y_condition = y.filter(mask)
        self.estimator_.fit(X_condition, y_condition)
        return self

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        """Predict using the estimator only where the condition is met."""
        self.check_input_types(X)

        predictions = np.full(X.height, self.default, dtype=np.float64)

        mask = X.select(self.condition).to_series()
        if self.verbose:
            print(f"Predicting on {mask.sum()}/{len(mask)} rows.")

        if not mask.any():
            return predictions

        X_condition = X.filter(mask)
        predictions[mask.to_numpy()] = self.estimator_.predict(X_condition)
        return predictions

    def check_input_types(self, X, y=None):
        if not isinstance(X, pl.DataFrame):
            raise TypeError("X must be a Polars DataFrame.")
        if y is not None and not isinstance(y, pl.Series):
            raise TypeError("y must be a Polars Series.")
