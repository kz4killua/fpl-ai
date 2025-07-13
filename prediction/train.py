import polars as pl

from datautil.load.merged import load_merged
from datautil.utils import get_seasons
from features.engineer_features import (
    engineer_match_features,
    engineer_player_features,
    engineer_team_features,
)
from prediction.model import PredictionModel
from prediction.utils import save_model


def train():
    # Load player and match data
    seasons = get_seasons("2023-24")
    players, teams, managers = load_merged(seasons)
    players = players.collect()
    teams = teams.collect()
    managers = managers.collect()

    # Engineer features for prediction
    players = engineer_player_features(players)
    teams = engineer_team_features(teams)
    matches = engineer_match_features(teams)

    # Fit and save models for simulations
    seasons = ["2021-22", "2022-23", "2023-24"]
    for season in seasons:
        # Only train on seasons before the current one
        players_season = players.filter(pl.col("season") < season)
        matches_season = matches.filter(pl.col("season") < season)
        model = PredictionModel()
        model.fit(players_season, matches_season)
        save_model(model, f"simulation_{season}")
