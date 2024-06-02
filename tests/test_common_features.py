import unittest

import pandas as pd
import numpy as np

from features.common.categorical import OneHotEncode
from features.common.exponential import ExponentialAverages
from features.common.condition import ConditionExponentialAverages
from features.common.deviation import StandardDeviation

from utilities import compare_arrays


class TestOneHotEncode(unittest.TestCase):

    def test_one_hot_encode(self):

        players = pd.DataFrame({
            'position': ['fwd', 'mid', 'def', 'gkp']
        })

        class PositionEncoder(OneHotEncode):
            COLUMN = 'position'
            
        encoder = PositionEncoder()

        features = encoder.engineer_features(players)

        self.assertTrue(
            (features['position_fwd'] == pd.Series([1, 0, 0, 0])).all()
        )
        self.assertTrue(
            (features['position_gkp'] == pd.Series([0, 0, 0, 1])).all()
        )


class TestExponentialAverages(unittest.TestCase):

    def setUp(self):
        self.players = pd.read_csv('tests/data/test_exponential_averages.csv')
        self.players['date'] = self.players['date'].apply(np.datetime64)


    def test_exponential_averages(self):

        class FeatureExtractor(ExponentialAverages):
            ITER_COLUMN = 'id'
            HALFLIFES = {'total_points': 10e3}
            DATE_COLUMN = 'date'

        features = FeatureExtractor().engineer_features(self.players)

        # Check correctness for player 1
        expected_1 = np.array([0, 3, 2, 1.33333, 1.33333]),
        obtained_1 = features[self.players['id'] == 1]['weighted_average_total_points_10000.0'].values
        self.assertTrue(compare_arrays(expected_1, obtained_1))

        # Check correctness for player 2
        expected_2 = np.array([0, 6, 3.5, 5.33333, 5.33333]),
        obtained_2 = features[self.players['id'] == 2]['weighted_average_total_points_10000.0'].values
        self.assertTrue(compare_arrays(expected_2, obtained_2))


    def test_condition_exponential_averages(self):

        class FeatureExtractor(ConditionExponentialAverages):
            ITER_COLUMN = 'id'
            HALFLIFES = {'total_points': 5}
            DATE_COLUMN = 'date'
            CONDITION_COLUMN = 'condition'

        features = FeatureExtractor().engineer_features(self.players)

        # Check correctness for player 1
        expected_1 = np.array([0.0, 0.0, 3.0, 1.0, 1.0])
        obtained_1 = features[self.players['id'] == 1][
            'condition_weighted_average_total_points_5'].values
        self.assertTrue(compare_arrays(expected_1, obtained_1))

        # Check correctness for player 2
        expected_2 = np.array([0, 0, 6, 1, 7.80750])
        obtained_2 = features[self.players['id'] == 2][
            'condition_weighted_average_total_points_5'].values
        self.assertTrue(compare_arrays(expected_2, obtained_2))


    def test_standard_deviation(self):

        class FeatureExtractor(StandardDeviation):
            ITER_COLUMN = 'id'
            WINDOWS = {'total_points': 1000}

            def get_feature_name(self, column, window):
                return f"std_{column}_{window}"

        features = FeatureExtractor().engineer_features(self.players)

        # Check correctness for player 1
        expected_1 = np.array([0, 0, 1.414214, 1.527525, 1.527525])
        obtained_1 = features[self.players['id'] == 1][
            'std_total_points_1000'].values
        self.assertTrue(compare_arrays(expected_1, obtained_1))

        # Check correctness for player 2
        expected_2 = np.array([0, 0, 3.535534, 4.041452, 4.041452])
        obtained_2 = features[self.players['id'] == 2][
            'std_total_points_1000'].values
        self.assertTrue(compare_arrays(expected_2, obtained_2))