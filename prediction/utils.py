import pickle
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import BaseCrossValidator

from game.rules import DEF, FWD, GKP, MID

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


class ElementTypeSplitEstimator(BaseEstimator):
    """Uses different estimators for different element types."""

    def __init__(self, estimator: BaseEstimator):
        self.estimator = estimator
        self.element_type_column = "element_type"

    def fit(self, x: pl.DataFrame, y: pl.Series):
        # Create a copy of the estimator for each element type
        self.gkp_estimator = clone(self.estimator)
        self.def_estimator = clone(self.estimator)
        self.mid_estimator = clone(self.estimator)
        self.fwd_estimator = clone(self.estimator)

        # Create masks for each element type
        gkps, defs, mids, fwds = self._get_element_type_masks(x)
        x = self._remove_element_type_column(x)

        # Fit each estimator on the corresponding element type
        self.gkp_estimator.fit(x.filter(gkps), y.filter(gkps))
        self.def_estimator.fit(x.filter(defs), y.filter(defs))
        self.mid_estimator.fit(x.filter(mids), y.filter(mids))
        self.fwd_estimator.fit(x.filter(fwds), y.filter(fwds))

        return self

    def predict(self, x: pl.DataFrame) -> np.ndarray:
        # Create an empty array to store the predictions
        y = np.zeros(len(x))

        # Create masks for each element type
        gkps, defs, mids, fwds = self._get_element_type_masks(x)
        x = self._remove_element_type_column(x)

        # Make predictions for each element type
        y[gkps.to_numpy()] = self.gkp_estimator.predict(x.filter(gkps))
        y[defs.to_numpy()] = self.def_estimator.predict(x.filter(defs))
        y[mids.to_numpy()] = self.mid_estimator.predict(x.filter(mids))
        y[fwds.to_numpy()] = self.fwd_estimator.predict(x.filter(fwds))

        return y

    def _get_element_type_masks(self, x: pl.DataFrame):
        """Returns masks for the different element types."""

        gkps = x[self.element_type_column] == GKP
        defs = x[self.element_type_column] == DEF
        mids = x[self.element_type_column] == MID
        fwds = x[self.element_type_column] == FWD

        if (gkps.sum() + defs.sum() + mids.sum() + fwds.sum()) != len(x):
            raise ValueError(
                "The input data does not contain the correct number of element types."
            )

        return gkps, defs, mids, fwds

    def _remove_element_type_column(self, x: pl.DataFrame) -> pl.DataFrame:
        """Removes the element type column from the data."""
        return x.drop(self.element_type_column)

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__class__.__name__
