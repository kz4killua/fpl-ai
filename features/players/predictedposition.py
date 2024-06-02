import pandas as pd

from features.base import FeatureEngineeringStep
from features.utilities import iter_players


class PredictedPlayerPosition(FeatureEngineeringStep):
    """Predicts each player's position in their next matches. """

    def engineer_features(self, players):

        last_positions = pd.Series(index=players.index, dtype=str)

        # Get the last understat position for each player
        for player in iter_players(players):
            last_positions.loc[player.index] = player['understat_position'].shift(1).fillna(method='ffill')

        # Fill in missing values
        last_positions = last_positions.fillna('Reserves')

        # Add to the dataframe
        players['last_position'] = last_positions