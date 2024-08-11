import warnings
import pickle
import json
from copy import deepcopy

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator

from datautil.utilities import GKP, DEF, MID, FWD


class PositionSplitEstimator(BaseEstimator):
    """Uses different estimators for different player positions."""

    def __init__(self, estimator):
        self.position_column = 'element_type'
        self.gkp_estimator = deepcopy(estimator)
        self.def_estimator = deepcopy(estimator)
        self.mid_estimator = deepcopy(estimator)
        self.fwd_estimator = deepcopy(estimator)


    def _get_position_masks(self, x):
        """Returns masks for the different player positions."""

        gkps = x[self.position_column] == GKP
        defs = x[self.position_column] == DEF
        mids = x[self.position_column] == MID
        fwds = x[self.position_column] == FWD

        if (gkps.sum() + defs.sum() + mids.sum() + fwds.sum()) != len(x):
            raise ValueError("The input data does not contain the correct number of player types.")

        return gkps, defs, mids, fwds


    def _remove_position_column(self, x):
        """Removes the position column from the data."""
        return x.drop(columns=self.position_column)


    def fit(self, x, y):
        gkps, defs, mids, fwds = self._get_position_masks(x)
        x = self._remove_position_column(x)
        self.gkp_estimator.fit(x[gkps], y[gkps])
        self.def_estimator.fit(x[defs], y[defs])
        self.mid_estimator.fit(x[mids], y[mids])
        self.fwd_estimator.fit(x[fwds], y[fwds])
        return self
    

    def predict(self, x):
        y = np.zeros(len(x))
        gkps, defs, mids, fwds = self._get_position_masks(x)
        x = self._remove_position_column(x)
        y[gkps] = self.gkp_estimator.predict(x[gkps])
        y[defs] = self.def_estimator.predict(x[defs])
        y[mids] = self.mid_estimator.predict(x[mids])
        y[fwds] = self.fwd_estimator.predict(x[fwds])
        return y


def make_predictions(features: pd.DataFrame, model_path: str, columns_path: str) -> pd.DataFrame:
    """Returns a dataframe mapping player IDs and fixture details to predicted points."""

    # Load the model and prediction columns
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(columns_path, 'r') as f:
        columns = json.load(f)

    if 'total_points' in columns:
        columns.remove('total_points')

    X = features[columns]

    if X.isna().sum().sum() > 0:
        warnings.warn("Missing data was found in the prediction features.")

    predictions = model.predict(X)

    # Map predicted points to player IDs, fixtures, and gameweeks
    predictions = pd.DataFrame({
        'element': features['element'].values,
        'fixture': features['fixture'].values,
        'round': features['round'].values,
        'total_points': predictions
    })

    return predictions


def group_predictions_by_gameweek(predictions: pd.DataFrame) -> pd.Series:
    """Sum up the number of points per player in each gameweek."""
    return predictions.groupby(['element', 'round']).sum()['total_points']


def weight_gameweek_predictions_by_availability(gameweek_predictions: pd.Series, elements: pd.DataFrame, next_gameweek: int):
    """Scale points predictions by a player's chance of playing."""

    gameweek_predictions = gameweek_predictions.copy()

    next_gameweek_availability = elements.set_index('id')['chance_of_playing_next_round'].fillna(100) / 100
    future_gameweek_availability = elements.set_index('id')['status'].replace({
        'a': 1, 'd': 1, 'i': 0, 'u': 0, 'n': 1, 's': 1
    }).astype('float')

    for gameweek_number in filter(lambda gameweek: gameweek >= next_gameweek, gameweek_predictions.index.get_level_values('round').unique()):
        
        # Weight the next week using 'chance_of_playing_next_round' and all others using 'status'
        if gameweek_number == next_gameweek:
            gameweek_number_availability = next_gameweek_availability
        else:
            gameweek_number_availability = future_gameweek_availability

        # Apply the weights to the predictions
        gameweek_number_predictions = gameweek_predictions.loc[:, gameweek_number]
        gameweek_number_predictions *= gameweek_number_availability[gameweek_number_predictions.index]
        
        # Update the predictions table
        gameweek_predictions.loc[:, gameweek_number] = gameweek_number_predictions.values

    return gameweek_predictions