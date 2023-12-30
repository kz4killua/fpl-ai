import pandas as pd
import numpy as np

from ..base import FeatureEngineeringStep
from ..utilities import iter_players, exponential_rolling


class PositionAveragesAgainstOpponent(FeatureEngineeringStep):
    """
    Calculate how well players in each position perform against the next opponent.
    """

    HALFLIFES = {
        'assists': 390,
        'bonus': 390,
        'bps': 390,
        'clean_sheets': 390,
        'creativity': 390,
        'goals_conceded': 390,
        'goals_scored': 390,
        'ict_index': 390,
        'influence': 340,
        'minutes': 390,
        'saves': 390,
        'threat': 390,
        'total_points': 390,
        'starts': 20,
        'expected_goals': 60,
        'expected_assists': 60,
        'expected_goal_involvements': 50,
        'expected_goals_conceded': 40,
        'understat_shots': 390,
        'understat_xG': 390,
        'understat_xA': 390,
        'understat_xGi': 390,
        'understat_key_passes': 390,
        'understat_xGChain': 390,
        'understat_xGBuildup': 390,
    }

    def engineer_features(self, players):

        output = pd.DataFrame(
            columns=list(self.HALFLIFES.keys()), dtype=float, index=players.index
        )

        # Sum up each position's performance against each opponent
        grouped = players.groupby(
            ['season', 'fixture', 'opponent_team_code', 'understat_position']
        )[list(self.HALFLIFES.keys())].sum()

        # Get all season-fixture combinations (in order of kickoff time)
        indices = pd.MultiIndex.from_product(
            [players['season'].unique(), players['fixture'].unique()], names=['season', 'fixture']
        )

        for column, halflife in self.HALFLIFES.items():

            averages = {}

            # Compute each opponent's average performance vs each position
            for opponent in players['opponent_team_code'].unique():
                for position in players['understat_position'].unique():

                    if position is np.nan:
                        continue

                    # Get data for relevant players
                    try:
                        series = grouped.loc[:, :, opponent, position][column]

                    # Skip opponents without previous fixtures
                    except KeyError:
                        continue

                    # Get kickoff times of all relevant players
                    times = players[
                        (players['opponent_team_code'] == opponent)
                        & (players['understat_position'] == position)
                    ]['kickoff_time'].drop_duplicates()

                    # Compute weighted averages over previous games
                    averages[opponent, position] = exponential_rolling(
                        series=series, times=times, halflife=pd.Timedelta(days=halflife), shift=0
                    )

                    # Re-index to cover all (season, fixture) combinations
                    averages[opponent, position] = averages[opponent, position].reindex(
                        indices
                    )

                    # Shift values by 1
                    averages[opponent, position] = averages[opponent, position].shift(1, fill_value=0)

                    # Fill in the gaps
                    averages[opponent, position] = averages[opponent, position].fillna(method='ffill').fillna(0)


            for opponent, position in averages.keys():
            
                # Pick out the relevant players (playing against opponent)
                subset = players[
                    (players['opponent_team_code'] == opponent) & (players['last_position'] == position)
                ]
        
                # Map in opponent position-specific performances
                output.loc[subset.index, column] = subset.set_index(['season', 'fixture']).index.map(
                    averages[opponent, position]
                )

        # Fill in missing values
        output = output.fillna(0)

        # Rename columns of the output dataframe
        output = output.rename(
            lambda c: f"opponent_position_average_{c}_{halflife}", axis=1
        )

        return output