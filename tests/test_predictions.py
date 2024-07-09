import unittest

import pandas as pd
import numpy as np

from predictions import group_predictions_by_gameweek, sum_player_points, weight_gameweek_predictions_by_availability


class TestPredictions(unittest.TestCase):

    def setUp(self):
        predictions = pd.read_csv('tests/data/test_sample_predictions.csv')
        self.gameweek_predictions = group_predictions_by_gameweek(predictions)
        self.elements = pd.read_csv('tests/data/test_elements.csv')


    def test_sum_player_points(self):

        test_cases = [
            {'players': [1, 2, 3], 'gameweek': 1, 'weights': 1, 'expected': 23},
            {'players': [1, 2, 3], 'gameweek': 2, 'weights': 1, 'expected': 7},
            {'players': [1, 2, 3], 'gameweek': 3, 'weights': 1, 'expected': 10},
            {'players': [1, 2, 3], 'gameweek': 4, 'weights': 1, 'expected': 16},
            {'players': [1, 2, 3], 'gameweek': 5, 'weights': 1, 'expected': 2},
            {'players': [2], 'gameweek': 1, 'weights': 1, 'expected': 13},
            {'players': [2], 'gameweek': 2, 'weights': 1, 'expected': 2},
            {'players': [2], 'gameweek': 3, 'weights': 1, 'expected': 5},
            {'players': [2], 'gameweek': 4, 'weights': 1, 'expected': 0},
            {'players': [2], 'gameweek': 5, 'weights': 1, 'expected': 0},
            {'players': [1, 2], 'gameweek': 4, 'weights': 1, 'expected': 8},
            {'players': [1, 2], 'gameweek': 5, 'weights': 1, 'expected': 1},
            {'players': [1, 2, 3], 'gameweek': 2, 'weights': 2, 'expected': 14},
            {'players': [1, 2, 3], 'gameweek': 2, 'weights': [0.5, 1.5, 2.5], 'expected': 7.5},
            {'players': [1, 2, 3], 'gameweek': 3, 'weights': [0.5, 1.5, 2.5], 'expected': 10},
        ]

        for test_case in test_cases:
            total_points = self.gameweek_predictions.loc[:, test_case['gameweek']].to_dict()
            self.assertEqual(
                sum_player_points(
                    test_case['players'], 
                    total_points, 
                    test_case['weights']
                ),
                test_case['expected']
            )


    def test_weight_gameweek_predictions_by_availability(self):
        
        weighted_predictions = weight_gameweek_predictions_by_availability(
            self.gameweek_predictions, self.elements, 1
        )

        test_cases = [
            {"element": 1, "round": 1, "expected": 5},
            {"element": 1, "round": 2, "expected": 4},
            {"element": 2, "round": 1, "expected": 9.75},
            {"element": 2, "round": 3, "expected": 5},
            {"element": 3, "round": 1, "expected": 0},
            {"element": 3, "round": 4, "expected": 0},
            {"element": 6, "round": 1, "expected": 5},
            {"element": 6, "round": 2, "expected": 5},
            {"element": 8, "round": 1, "expected": 0},
            {"element": 8, "round": 3, "expected": 2},
            {"element": 11, "round": 1, "expected": 11},
            {"element": 15, "round": 3, "expected": 0},
        ]

        for test_case in test_cases:
            self.assertAlmostEqual(
                weighted_predictions.loc[test_case['element'], test_case['round']], test_case['expected']
        )