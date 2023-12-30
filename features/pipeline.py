import pandas as pd

from tqdm import tqdm


class FeaturePipeline:

    STEPS = [

    ]

    def apply(self, df):
        """
        Apply a feature engineering pipeline, then return created features.
        """

        columns = []

        for step in tqdm(self.STEPS, desc="Engineering features"):

            # Apply each feature engineering step
            features = step.engineer_features(df)

            if features is not None:
                # Add new features to the df
                df = pd.concat([df, features], axis=1)

                # Keep track of newly created features
                columns.extend(features.columns)

        return df, columns
