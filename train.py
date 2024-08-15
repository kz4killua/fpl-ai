import random
import pickle
import numpy as np
import json
from datautil.pipeline import load_players_and_teams
from features.features import engineer_features
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from predictions import PositionSplitEstimator


RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def get_model():
    pipeline = make_pipeline(
        StandardScaler(),
        LinearRegression()
    )
    return PositionSplitEstimator(pipeline)


def load_features(seasons):
    local_players, local_teams = load_players_and_teams(seasons)
    features, columns = engineer_features(local_players, local_teams)
    return features, columns


def train_model(features, columns):
    x = features[columns].drop(columns=['total_points'])
    y = features['total_points']
    model = get_model()
    model.fit(x, y)
    return model


def save_model(model, columns, tag):
    with open(f'model-{tag}.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open(f'columns-{tag}.json', 'w') as f:
        json.dump(columns, f)


def main():

    seasons = ['2016-17', '2017-18', '2018-19', '2019-20', '2020-21', '2021-22', '2022-23', '2023-24']

    # Train one model on all seasons
    features, columns = load_features(seasons)
    model = train_model(features, columns)
    save_model(model, columns, 'all')

    # Train other models, excluding one season at a time (This is useful for simulations)
    for excluded_season in ['2021-22', '2022-23', '2023-24']:
        excluded_features = features[features['season'] != excluded_season]
        excluded_columns = columns.copy()
        excluded_model = train_model(excluded_features, excluded_columns)
        save_model(excluded_model, excluded_columns, f'excluded-{excluded_season}')


if __name__ == '__main__':
    main()