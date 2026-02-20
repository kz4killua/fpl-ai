from collections.abc import Iterable

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator

from .assists import make_assists_predictor
from .bonus import make_bonus_predictor
from .bps import make_bps_predictor
from .clean_sheets import make_clean_sheets_predictor
from .clearances_blocks_interceptions import (
    make_clearances_blocks_interceptions_predictor,
)
from .defensive_contribution import (
    calculate_defensive_contribution_threshold_prob,
    make_defensive_contribution_predictor,
)
from .goals_conceded import make_goals_conceded_predictor
from .goals_scored import make_goals_scored_predictor
from .minutes import make_minutes_category_predictor
from .recoveries import make_recoveries_predictor
from .saves import make_saves_predictor
from .tackles import make_tackles_predictor
from .team_goals_scored import make_team_goals_scored_predictor
from .total_points import make_total_points_predictor


class PredictionModel:
    def __init__(self):
        self.team_goals_scored_predictor = make_team_goals_scored_predictor()
        self.minutes_category_predictor = make_minutes_category_predictor()
        self.clean_sheets_predictor = make_clean_sheets_predictor()
        self.goals_conceded_predictor = make_goals_conceded_predictor()
        self.goals_scored_predictor = make_goals_scored_predictor()
        self.assists_predictor = make_assists_predictor()
        self.saves_predictor = make_saves_predictor()
        self.bps_predictor = make_bps_predictor()
        self.bonus_predictor = make_bonus_predictor()
        self.clearances_blocks_interceptions_predictor = (
            make_clearances_blocks_interceptions_predictor()
        )
        self.tackles_predictor = make_tackles_predictor()
        self.recoveries_predictor = make_recoveries_predictor()
        self.defensive_contribution_predictor = make_defensive_contribution_predictor()
        self.total_points_predictor = make_total_points_predictor()

        self.player_pipeline: list[GenericPipelineStep] = [
            MinutesCategoryPipelineStep(
                self.minutes_category_predictor,
                "minutes_category",
            ),
            GenericPipelineStep(
                self.goals_scored_predictor,
                "goals_scored",
            ),
            GenericPipelineStep(
                self.assists_predictor,
                "assists",
            ),
            GenericPipelineStep(
                self.clean_sheets_predictor,
                "clean_sheets",
            ),
            GenericPipelineStep(
                self.goals_conceded_predictor,
                "goals_conceded",
            ),
            GenericPipelineStep(
                self.saves_predictor,
                "saves",
            ),
            GenericPipelineStep(
                self.clearances_blocks_interceptions_predictor,
                "clearances_blocks_interceptions",
            ),
            GenericPipelineStep(
                self.tackles_predictor,
                "tackles",
            ),
            GenericPipelineStep(
                self.recoveries_predictor,
                "recoveries",
            ),
            GenericPipelineStep(
                self.defensive_contribution_predictor,
                "defensive_contribution",
            ),
            DefensiveContributionThresholdProbStep(),
            GenericPipelineStep(
                self.bps_predictor,
                "bps",
            ),
            GenericPipelineStep(
                self.bonus_predictor,
                "bonus",
            ),
            GenericPipelineStep(
                self.total_points_predictor,
                "total_points",
            ),
        ]

    def fit(self, players: pl.DataFrame, matches: pl.DataFrame):
        """Fit all sub-models in order."""

        # Fit the team goals model and predict match results
        fit_model(
            self.team_goals_scored_predictor,
            matches,
            ["team_h_goals_scored", "team_a_goals_scored"],
        )
        predicted_match_results = predict_match_results(
            self.team_goals_scored_predictor, matches
        )
        X = combine_match_results(players, matches, predicted_match_results)

        # Fit each step in the player pipeline
        for step in self.player_pipeline:
            step.fit(X)
            predictions = step.predict(X)
            X = step.combine(X, predictions)

        return self

    def predict(
        self,
        players: pl.DataFrame,
        matches: pl.DataFrame,
        return_dataframe: bool = False,
    ) -> pl.Series | pl.DataFrame:
        """Predict total points for each player in each fixture."""

        # Predict match results and combine with player data
        predicted_match_results = predict_match_results(
            self.team_goals_scored_predictor, matches
        )
        X = combine_match_results(players, matches, predicted_match_results)

        # Run each step in the player pipeline
        for step in self.player_pipeline:
            predictions = step.predict(X)
            X = step.combine(X, predictions)

        if not return_dataframe:
            return X.get_column("predicted_total_points")
        return X


class GenericPipelineStep:
    def __init__(self, model: BaseEstimator, target: str):
        self.model = model
        self.target = target

    def fit(self, X: pl.DataFrame):
        return fit_model(self.model, X, self.target)

    def predict(self, X: pl.DataFrame) -> pl.Series:
        predictions = self.model.predict(X)
        return pl.Series(f"predicted_{self.target}", predictions, dtype=pl.Float64)

    def combine(self, X: pl.DataFrame, predictions: pl.Series | pl.DataFrame):
        if isinstance(predictions, pl.DataFrame):
            return pl.concat([X, predictions], how="horizontal")
        else:
            return X.with_columns(predictions)


class MinutesCategoryPipelineStep(GenericPipelineStep):
    def predict(self, X: pl.DataFrame) -> pl.DataFrame:
        # Predict probabilities for each class
        predictions = self.model.predict_proba(X)
        classes = list(self.model.named_steps["predictor"].classes_)
        classes = [f"predicted_{c}" for c in classes]
        return pl.DataFrame(predictions, schema=classes)


class DefensiveContributionThresholdProbStep(GenericPipelineStep):
    def __init__(self):
        pass

    def fit(self, X: pl.DataFrame):
        return self

    def predict(self, X: pl.DataFrame) -> pl.Series:
        probs = calculate_defensive_contribution_threshold_prob(
            X, "predicted_defensive_contribution"
        )
        return pl.Series(
            "predicted_defensive_contribution_threshold_prob", probs, dtype=pl.Float64
        )

    def combine(self, X: pl.DataFrame, predictions: pl.Series):
        return X.with_columns(predictions)


def combine_match_results(
    players: pl.DataFrame,
    matches: pl.DataFrame,
    predicted_match_results: pl.DataFrame,
):
    # Merge the predicted results with the matches DataFrame
    matches = pl.concat([matches, predicted_match_results], how="horizontal")

    # Join the predicted results with the players DataFrame
    players = players.join(
        matches.select(
            [
                pl.col("season"),
                pl.col("fixture_id").alias("fixture"),
                pl.col("predicted_team_h_goals_scored"),
                pl.col("predicted_team_a_goals_scored"),
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
            .then(pl.col("predicted_team_h_goals_scored"))
            .otherwise(pl.col("predicted_team_a_goals_scored"))
            .alias("predicted_team_goals_scored"),
            pl.when(pl.col("was_home") == 1)
            .then(pl.col("predicted_team_h_clean_sheets"))
            .otherwise(pl.col("predicted_team_a_clean_sheets"))
            .alias("predicted_team_clean_sheets"),
            pl.when(pl.col("was_home") == 1)
            .then(pl.col("predicted_team_a_goals_scored"))
            .otherwise(pl.col("predicted_team_h_goals_scored"))
            .alias("predicted_opponent_goals_scored"),
            pl.when(pl.col("was_home") == 1)
            .then(pl.col("predicted_team_a_clean_sheets"))
            .otherwise(pl.col("predicted_team_h_clean_sheets"))
            .alias("predicted_opponent_clean_sheets"),
        ]
    )
    players = players.drop(
        [
            "predicted_team_h_goals_scored",
            "predicted_team_a_goals_scored",
            "predicted_team_h_clean_sheets",
            "predicted_team_a_clean_sheets",
        ]
    )
    return players


def predict_match_results(goals_predictor: BaseEstimator, matches: pl.DataFrame):
    """Predict team goals scored and clean sheets for each match."""
    predictions = goals_predictor.predict(matches)
    predicted_team_h_goals_scored = predictions[:, 0]
    predicted_team_a_goals_scored = predictions[:, 1]

    # Predict clean sheets assuming a Poisson distribution for goals
    predicted_team_h_clean_sheets = np.exp(-predicted_team_a_goals_scored)
    predicted_team_a_clean_sheets = np.exp(-predicted_team_h_goals_scored)

    return pl.DataFrame(
        {
            "predicted_team_h_goals_scored": predicted_team_h_goals_scored,
            "predicted_team_a_goals_scored": predicted_team_a_goals_scored,
            "predicted_team_h_clean_sheets": predicted_team_h_clean_sheets,
            "predicted_team_a_clean_sheets": predicted_team_a_clean_sheets,
        }
    )


def fit_model(model: BaseEstimator, X: pl.DataFrame, target: str | Iterable[str]):
    X = X.drop_nulls(subset=target)
    X_fit = X.drop(target)
    y_fit = X.get_column(target) if isinstance(target, str) else X.select(target)
    return model.fit(X_fit, y_fit)
