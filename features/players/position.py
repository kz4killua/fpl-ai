import itertools
import pandas as pd
import numpy as np

from ..base import FeatureEngineeringStep
from ..utilities import exponential_rolling


class PositionAveragesAgainstOpponent(FeatureEngineeringStep):
    """
    Calculate how well players in each position perform against the next opponent.

    For each player, this answers the question: 
    "How well do other players in my position perform against this opponent, on average?"
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
        'understat_shots': 390,
        'understat_xG': 390,
        'understat_xA': 390,
        'understat_xGi': 390,
        'understat_key_passes': 390,
        'understat_xGChain': 390,
        'understat_xGBuildup': 390,
    }


    def engineer_features(self, players: pd.DataFrame) -> pd.DataFrame:

        if not players['kickoff_time'].is_monotonic_increasing:
            raise ValueError("Kickoff times must be in increasing order.")

        output = pd.DataFrame(
            columns=list(self.HALFLIFES.keys()), 
            dtype=float, 
            index=players.index
        )

        # Sum up each position's performance against each opponent (per fixture)
        grouped = players.groupby(
            ['season', 'fixture', 'opponent_team_code', 'understat_position'],
            sort=False
        )[list(self.HALFLIFES.keys())].sum()

        # Store the kickoff times for each season-fixture combination
        kickoff_times = players.set_index(['season', 'fixture'])['kickoff_time']
        kickoff_times = kickoff_times[~kickoff_times.index.duplicated()]

        # Get all season-fixture combinations
        indices = pd.MultiIndex.from_product(
            [players['season'].unique(), players['fixture'].unique()], 
            names=['season', 'fixture']
        )

        features = dict()


        for column, halflife in self.HALFLIFES.items():

            for opponent, position in itertools.product(
                players['opponent_team_code'].unique(),
                players['understat_position'].unique()
            ):
                
                # Ignore players with no understat.com records
                if (position is np.nan):
                    continue

                # Ignore invalid (opponent, position) combinations
                try:
                    series = grouped.loc[:, :, opponent, position][column]
                except KeyError:
                    continue

                # Get kickoff times for (opponent, position) combination
                times = kickoff_times.loc[series.index]

                # Compute weighted averages, indexed by (season, fixture)
                averages = exponential_rolling(
                    series=series, times=times, halflife=pd.Timedelta(days=halflife), shift=0
                )
                averages = averages.reindex(indices)
                averages = averages.shift(1, fill_value=0)
                averages = averages.ffill()
                averages = averages.fillna(0)
                averages = averages.to_dict()

                features[column, opponent, position] = averages


        def mapper(row: pd.Series, column: str) -> float:
            """Retrieve a specific average from the features dictionary."""
            opponent = row['opponent_team_code']
            position = row['last_position']
            season = row['season']
            fixture = row['fixture']

            # Return 0 if the average does not exist
            try:
                averages = features[column, opponent, position]
            except KeyError:
                return 0

            return averages[season, fixture]
        

        # Finally, map the averages to the appropriate players
        for column, halflife in self.HALFLIFES.items():
            output[column] = players.apply(lambda row: mapper(row, column), axis=1)

        # Rename columns of the output dataframe
        output = output.rename(
            lambda c: f"opponent_position_average_{c}_{self.HALFLIFES[c]}", axis=1
        )

        return output