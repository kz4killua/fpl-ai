import warnings

import numpy as np
import polars as pl

from .match import make_match_predictor
from .minutes import make_minutes_predictor
from .total_points import make_total_points_predictor


class PredictionModel:
    def __init__(self):
        # Create the intermediate predictors
        self.minutes_predictor = make_minutes_predictor()
        self.match_predictor = make_match_predictor()
        self.total_points_predictor = make_total_points_predictor()

    def fit(self, players: pl.DataFrame, matches: pl.DataFrame):
        """Fit the model to predict total points."""
        # Fit the intermediate models
        self._fit_minutes(players)
        self._fit_matches(matches)
        # Make intermediate predictions
        predicted_minutes = self._predict_minutes(players)
        predicted_results = self._predict_matches(matches)
        # Fit the final model
        X = self._prepare_features(
            players, matches, predicted_minutes, predicted_results
        )
        self._fit_total_points(X)
        return self

    def predict(self, players: pl.DataFrame, matches: pl.DataFrame):
        """Predict total points for each player in each fixture."""
        # Make intermediate predictions
        predicted_minutes = self._predict_minutes(players)
        predicted_results = self._predict_matches(matches)
        X = self._prepare_features(
            players, matches, predicted_minutes, predicted_results
        )
        # Make final predictions
        return self._predict_total_points(X)

    def _prepare_features(
        self,
        players: pl.DataFrame,
        matches: pl.DataFrame,
        predicted_minutes: pl.Series,
        predicted_results: pl.DataFrame,
    ):
        # Create dataframe with the features for the final model
        players = players.with_columns(predicted_minutes)
        matches = pl.concat([matches, predicted_results], how="horizontal")
        players = players.join(
            matches.select(
                [
                    pl.col("season"),
                    pl.col("fixture_id").alias("fixture"),
                    pl.col("predicted_team_h_scored"),
                    pl.col("predicted_team_a_scored"),
                    pl.col("predicted_team_h_clean_sheets"),
                    pl.col("predicted_team_a_clean_sheets"),
                ]
            ),
            on=["season", "fixture"],
            how="left",
        )
        players = players.with_columns(
            [
                pl.when(pl.col("was_home") == 1)
                .then(pl.col("predicted_team_h_scored"))
                .otherwise(pl.col("predicted_team_a_scored"))
                .alias("predicted_team_scored"),
                pl.when(pl.col("was_home") == 1)
                .then(pl.col("predicted_team_h_clean_sheets"))
                .otherwise(pl.col("predicted_team_a_clean_sheets"))
                .alias("predicted_team_clean_sheets"),
                pl.when(pl.col("was_home") == 1)
                .then(pl.col("predicted_team_a_scored"))
                .otherwise(pl.col("predicted_team_h_scored"))
                .alias("predicted_opponent_scored"),
                pl.when(pl.col("was_home") == 1)
                .then(pl.col("predicted_team_a_clean_sheets"))
                .otherwise(pl.col("predicted_team_h_clean_sheets"))
                .alias("predicted_opponent_clean_sheets"),
            ]
        )
        players = players.drop(
            [
                "predicted_team_h_scored",
                "predicted_team_a_scored",
                "predicted_team_h_clean_sheets",
                "predicted_team_a_clean_sheets",
            ]
        )
        # Check that the merge was successful
        self._check_features(players)
        return players

    def _check_features(self, players: pl.DataFrame):
        if (
            players.filter(
                pl.col("predicted_minutes").is_null()
                | pl.col("predicted_team_scored").is_null()
                | pl.col("predicted_opponent_scored").is_null()
                | pl.col("predicted_team_clean_sheets").is_null()
                | pl.col("predicted_opponent_clean_sheets").is_null()
            ).height
            > 0
        ):
            warnings.warn(
                "Some players have missing intermediate predictions.", stacklevel=2
            )

    def _fit_minutes(self, players: pl.DataFrame):
        target = "minutes"
        X = players.drop(target)
        y = players.get_column(target)
        self.minutes_predictor.fit(X, y)

    def _fit_matches(self, matches: pl.DataFrame):
        targets = ["team_h_scored", "team_a_scored"]
        X = matches.drop(targets)
        y = matches.select(targets)
        self.match_predictor.fit(X, y)

    def _fit_total_points(self, players: pl.DataFrame):
        target = "total_points"
        X = players.drop(target)
        y = players.get_column(target)
        self.total_points_predictor.fit(X, y)

    def _predict_minutes(self, players: pl.DataFrame):
        predictions = self.minutes_predictor.predict(players)
        return pl.Series("predicted_minutes", predictions, dtype=pl.Float64)

    def _predict_matches(self, matches: pl.DataFrame):
        # Make predictions for home and away goals
        predictions = self.match_predictor.predict(matches)
        predicted_team_h_scored = predictions[:, 0]
        predicted_team_a_scored = predictions[:, 1]
        # Predict clean sheets assuming a Poisson distribution for goals
        predicted_team_h_clean_sheets = np.exp(-predicted_team_a_scored)
        predicted_team_a_clean_sheets = np.exp(-predicted_team_h_scored)
        # Return predictions as a DataFrame
        return pl.DataFrame(
            {
                "predicted_team_h_scored": predicted_team_h_scored,
                "predicted_team_a_scored": predicted_team_a_scored,
                "predicted_team_h_clean_sheets": predicted_team_h_clean_sheets,
                "predicted_team_a_clean_sheets": predicted_team_a_clean_sheets,
            }
        )

    def _predict_total_points(self, players: pl.DataFrame):
        predictions = self.total_points_predictor.predict(players)
        return pl.Series("predicted_total_points", predictions, dtype=pl.Float64)
