import pandas as pd

from ..base import FeatureEngineeringStep
from ..utilities import iter_column, exponential_rolling


class ExponentialAverages(FeatureEngineeringStep):
    """Compute exponential weighted averages."""

    ITER_COLUMN = None
    
    HALFLIFES  = None

    DATE_COLUMN = None

    def engineer_features(self, df):

        output = pd.DataFrame(dtype=float, index=df.index)

        for column, halflife in self.HALFLIFES.items():
            for subset in iter_column(df, self.ITER_COLUMN):
                output.loc[subset.index, f'weighted_average_{column}_{halflife}'] = exponential_rolling(
                    subset[column], times=subset[self.DATE_COLUMN], halflife=pd.Timedelta(days=halflife)
                )

        return output