import polars as pl

from features.engineer_features import engineer_match_features, engineer_player_features
from loaders.merged import load_merged
from loaders.utils import force_dataframe, get_seasons
from prediction.model import PredictionModel
from prediction.utils import save_model


def train():
    # Load player and match data
    seasons = get_seasons(2024)
    players, matches, _ = load_merged(seasons)
    players = players.collect()
    matches = matches.collect()

    # Engineer features for prediction
    players = engineer_player_features(players)
    matches = engineer_match_features(matches)

    # Collect lazyframes
    players = force_dataframe(players)
    matches = force_dataframe(matches)

    # Fit and save models for simulations
    for season in seasons:
        if season < 2021:
            continue

        # Only train on seasons before the current one
        filtered_players = players.filter(pl.col("season") < season)
        filtered_matches = matches.filter(pl.col("season") < season)
        model = PredictionModel()
        model.fit(filtered_players, filtered_matches)
        save_model(model, f"simulation_{season}")

    # Fit a final model on all data
    model = PredictionModel()
    model.fit(players, matches)
    save_model(model, "live")
