import pandas as pd

from ..base import FeatureEngineeringStep


class OneHotEncode(FeatureEngineeringStep):

    COLUMN = None

    def engineer_features(self, df):
        """One-hot-encode a column of a dataframe."""

        # Get one-hot columns
        dummies = pd.get_dummies(df[self.COLUMN])

        # Rename appropriately
        dummies = dummies.rename(lambda name: f"{self.COLUMN}_{name}", axis=1)

        return dummies