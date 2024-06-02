import unittest

import pandas as pd
import numpy as np

from features.players.position import PositionAveragesAgainstOpponent
from features.players.predictedposition import PredictedPlayerPosition
from features.players.xgi import UnderstatXGI

from utilities import compare_arrays


class TestPositionAveragesAgainstOpponent(unittest.TestCase):

    def setUp(self):
        self.players = pd.read_csv('tests/data/test_position_averages.csv')
        self.players['kickoff_time'] = self.players['kickoff_time'].apply(np.datetime64)


    def test_predicted_player_position(self):
        
        PredictedPlayerPosition().engineer_features(self.players)

        expected_1 = ['FWD', 'DEF', 'FWD', 'MID', 'Reserves']
        obtained_1 = list(self.players['last_position'].values[-5:])
        self.assertSequenceEqual(expected_1, obtained_1)

        expected_2 = ['Reserves', 'FWD', 'FWD', 'FWD', 'FWD', 'DEF']
        obtained_2 = list(self.players[self.players['code'] == 2]['last_position'].values)
        self.assertSequenceEqual(expected_2, obtained_2)


    def test_position_averages_against_opponent(self):
        
        PredictedPlayerPosition().engineer_features(self.players)
        
        class FeatureExtractor(PositionAveragesAgainstOpponent):
            HALFLIFES = {'total_points': 1}

        features = FeatureExtractor().engineer_features(self.players)

        # Check correctness of output
        expected = np.array([0.092715, 0, 1, 2.967833, 0])
        obtained = features['opponent_position_average_total_points_1'].values[-5:]
        self.assertTrue(compare_arrays(expected, obtained))


class TestUnderstatXGI(unittest.TestCase):

    def setUp(self):
        self.players = pd.read_csv('tests/data/test_player_xgi.csv')

    def test_understat_xgi(self):

        UnderstatXGI().engineer_features(self.players)

        # Check correctness of output
        expected = np.array([4.55, 3.41, 1.7, 3.13, 5.8, 6.42, 2.47, 4.32, 3.42, 5.3])
        obtained = self.players['understat_xGi'].values
        self.assertTrue(compare_arrays(expected, obtained))
