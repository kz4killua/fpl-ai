import pandas as pd
import numpy as np

from ..base import FeatureEngineeringStep


class UnderstatXGI(FeatureEngineeringStep):
    """Compute expected goal involvement using understat.com data."""

    def engineer_features(self, df):
        
        df['understat_xGi'] = df['understat_xG'] + df['understat_xA']

        return None