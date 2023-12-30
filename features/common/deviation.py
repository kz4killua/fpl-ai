from ..base import FeatureEngineeringStep
from ..utilities import iter_column, simple_rolling

import pandas as pd


class StandardDeviation(FeatureEngineeringStep):

    ITER_COLUMN = None
    
    WINDOWS = None

    def engineer_features(self, df):
        
        output = pd.DataFrame(
            dtype=float, index=df.index
        )

        for column, window in self.WINDOWS.items():
            for team in iter_column(df, self.ITER_COLUMN):
                output.loc[team.index, self.get_feature_name(column, window)] = simple_rolling(
                    team[column], window, operation='std'
                )

        return output


    def get_feature_name(self, column, window):
        """
        Legacy: Get the name for a column computing std over a given window.
        """
        raise NotImplementedError