import warnings
import pickle
import json
from collections.abc import Iterable

import pandas as pd


def make_predictions(features: pd.DataFrame, model_path: str, columns_path: str) -> pd.DataFrame:
    """
    Returns a dataframe mapping player IDs and fixture details to predicted points.
    """

    # Load the model and prediction columns
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(columns_path, 'r') as f:
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


def sum_player_points(players: list, total_points: dict, weights: float | Iterable[float] = 1) -> float:
    """
    Add up (and optionally, weight) the total points for a list of players.
    """

    # Note: The following approach is faster than a vectorized approach
    points = 0

    # Weights should be an iterable of numeric values
    if not isinstance(weights, Iterable):
        weights = [weights for i in range(len(players))]

    # Sum up points for each player
    for i, element in enumerate(players):
        points += total_points.get(element, 0) * weights[i]

    return points


def weight_gameweek_predictions_by_availability(gameweek_predictions: pd.Series, elements: pd.DataFrame, next_gameweek: int):
    """
    Scale points predictions by a player's chance of playing.
    """

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