import warnings
from collections.abc import Iterable

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator

from .assists import make_assists_predictor
from .bps import make_bps_predictor
from .goals_scored import make_goals_scored_predictor
from .match import make_match_predictor
from .minutes import make_minutes_predictor
from .saves import make_saves_predictor
from .total_points import make_total_points_predictor


class PredictionModel:
    def __init__(self):
        self.minutes_predictor = make_minutes_predictor()
        self.match_predictor = make_match_predictor()
        self.goals_scored_predictor = make_goals_scored_predictor()
        self.assists_predictor = make_assists_predictor()
        self.saves_predictor = make_saves_predictor()
        self.bps_predictor = make_bps_predictor()
        self.total_points_predictor = make_total_points_predictor()

    def fit(self, players: pl.DataFrame, matches: pl.DataFrame):
        """Fit the model to predict total points."""
        # Fit models for minutes and match results
        self._fit_minutes(players)
        self._fit_matches(matches)
        predicted_minutes = self._predict_minutes(players)
        predicted_results = self._predict_matches(matches)
        X = self._merge_minutes(players, predicted_minutes)
        X = self._merge_matches(X, matches, predicted_results)
        # Fit models for goals, assists, and saves
        self._fit_goals_scored(X)
        self._fit_assists(X)
        self._fit_saves(X)
        predicted_goals_scored = self._predict_goals_scored(X)
        predicted_assists = self._predict_assists(X)
        predicted_saves = self._predict_saves(X)
        X = self._merge_goals_scored(X, predicted_goals_scored)
        X = self._merge_assists(X, predicted_assists)
        X = self._merge_saves(X, predicted_saves)
        # Fit the bonus points model
        self._fit_bps(X)
        predicted_bps = self._predict_bps(X)
        X = self._merge_bps(X, predicted_bps)
        # Fit the total points model
        self._fit_total_points(X)
        return self

    def predict(self, players: pl.DataFrame, matches: pl.DataFrame):
        """Predict total points for each player in each fixture."""
        # Predict minutes and match results
        predicted_minutes = self._predict_minutes(players)
        predicted_results = self._predict_matches(matches)
        X = self._merge_minutes(players, predicted_minutes)
        X = self._merge_matches(X, matches, predicted_results)
        # Predict goals, assists, and saves
        predicted_goals_scored = self._predict_goals_scored(X)
        predicted_assists = self._predict_assists(X)
        predicted_saves = self._predict_saves(X)
        X = self._merge_goals_scored(X, predicted_goals_scored)
        X = self._merge_assists(X, predicted_assists)
        X = self._merge_saves(X, predicted_saves)
        # Predict bonus points
        predicted_bps = self._predict_bps(X)
        X = self._merge_bps(X, predicted_bps)
        # Predict total points
        return self._predict_total_points(X)

    def _fit_minutes(self, players: pl.DataFrame):
        return self._fit_model(self.minutes_predictor, players, "minutes")

    def _fit_matches(self, matches: pl.DataFrame):
        return self._fit_model(
            self.match_predictor, matches, ["team_h_scored", "team_a_scored"]
        )

    def _fit_goals_scored(self, players: pl.DataFrame):
        return self._fit_model(self.goals_scored_predictor, players, "goals_scored")

    def _fit_assists(self, players: pl.DataFrame):
        return self._fit_model(self.assists_predictor, players, "assists")

    def _fit_saves(self, players: pl.DataFrame):
        return self._fit_model(self.saves_predictor, players, "saves")

    def _fit_bps(self, players: pl.DataFrame):
        return self._fit_model(self.bps_predictor, players, "bps")

    def _fit_total_points(self, players: pl.DataFrame):
        return self._fit_model(self.total_points_predictor, players, "total_points")

    def _fit_model(
        self, model: BaseEstimator, df: pl.DataFrame, target: str | Iterable[str]
    ):
        """Fit a model on one or more target columns."""
        X = df.drop(target)
        y = df.get_column(target) if isinstance(target, str) else df.select(target)
        return model.fit(X, y)

    def _merge_minutes(self, players: pl.DataFrame, predicted_minutes: pl.Series):
        return players.with_columns(predicted_minutes)

    def _merge_matches(
        self,
        players: pl.DataFrame,
        matches: pl.DataFrame,
        predicted_results: pl.DataFrame,
    ):
        # Merge the predicted results with the matches DataFrame
        matches = pl.concat([matches, predicted_results], how="horizontal")
        # Join the predicted results with the players DataFrame
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
        if (
            players.filter(
                pl.col("predicted_team_scored").is_null()
                | pl.col("predicted_opponent_scored").is_null()
                | pl.col("predicted_team_clean_sheets").is_null()
                | pl.col("predicted_opponent_clean_sheets").is_null()
            ).height
            > 0
        ):
            warnings.warn(
                "Some players have missing match result predictions.", stacklevel=2
            )

        return players

    def _merge_goals_scored(
        self, players: pl.DataFrame, predicted_goals_scored: pl.Series
    ):
        return players.with_columns(predicted_goals_scored)

    def _merge_assists(self, players: pl.DataFrame, predicted_assists: pl.Series):
        return players.with_columns(predicted_assists)

    def _merge_saves(self, players: pl.DataFrame, predicted_saves: pl.Series):
        return players.with_columns(predicted_saves)

    def _merge_bps(self, players: pl.DataFrame, predicted_bps: pl.Series):
        players = players.with_columns(predicted_bps)
        # Add BPS ranks
        players = players.with_columns(
            pl.col("predicted_bps")
            .rank(method="dense", descending=True)
            .over(["season", "fixture"])
            .alias("predicted_bps_rank")
        )
        return players

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

    def _predict_goals_scored(self, players: pl.DataFrame):
        predictions = self.goals_scored_predictor.predict(players)
        return pl.Series("predicted_goals_scored", predictions, dtype=pl.Float64)

    def _predict_assists(self, players: pl.DataFrame):
        predictions = self.assists_predictor.predict(players)
        return pl.Series("predicted_assists", predictions, dtype=pl.Float64)

    def _predict_saves(self, players: pl.DataFrame):
        predictions = self.saves_predictor.predict(players)
        return pl.Series("predicted_saves", predictions, dtype=pl.Float64)

    def _predict_bps(self, players: pl.DataFrame):
        predictions = self.bps_predictor.predict(players)
        return pl.Series("predicted_bps", predictions, dtype=pl.Float64)

    def _predict_total_points(self, players: pl.DataFrame):
        predictions = self.total_points_predictor.predict(players)
        return pl.Series("predicted_total_points", predictions, dtype=pl.Float64)
