import warnings
from typing import Union, Iterable
import pickle
import json
from collections import abc

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


def sum_player_points(players: list, total_points: pd.Series, weights: Union[None, float, Iterable[float]] = None) -> float:
    """
    Add up (and optionally, weight) the total points for a list of players.
    """
    
    points = 0

    # Weights should be an iterable of numeric values
    if not isinstance(weights, abc.Iterable):
        if weights is None:
            weights = [1 for i in range(len(players))]
        else:
            weights = [weights for i in range(len(players))]
    
    # Sum up points for each player
    for i, element in enumerate(players):
        points += total_points.get(element, 0) * weights[i]

    return points


def weight_gameweek_predictions_by_availability(gameweek_predictions: pd.Series, elements: pd.DataFrame, next_gameweek: int):
    """
    Scale points predictions by a player's chance of playing.
    """

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