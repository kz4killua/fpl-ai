import warnings
from typing import Union, Iterable
import pickle
import json

import pandas as pd
import numpy as np


def make_predictions(features: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a dataframe mapping player IDs and fixture details to predicted points.
    """

    # Load the model and prediction columns
    with open('models/model-all/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/model-all/columns.json') as f:
        columns = json.load(f)

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
    """
    Sum up the number of points per player in each gameweek.
    """
    return predictions.groupby(['element', 'round']).sum()['total_points']


def sum_gameweek_predictions(players: list, gameweek: int, gameweek_predictions: pd.Series, weights: Union[None, float, Iterable[float]] = None) -> float:
    """
    Add up (optionally weighted) predicted points for a list of players in a gameweek.
    """
    
    try:
        # Retrieve predictions (default value of 0) for each player
        player_predictions = pd.Series({
            (element, gameweek): 0 for element in players
        })
        player_predictions.update(
            gameweek_predictions.loc[players, gameweek]
        )

        # Apply weights to predicted points if provided.
        if weights is not None:
            player_predictions *= weights

        # Sum up total points
        return player_predictions.sum()

    # Return 0 points when a player has no fixtures in the gameweek.
    except KeyError:
        return 0