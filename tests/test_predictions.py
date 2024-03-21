import unittest

import pandas as pd
import numpy as np

from predictions import group_predictions_by_gameweek, sum_gameweek_predictions, weight_gameweek_predictions_by_availability


class TestPredictions(unittest.TestCase):

    def setUp(self):
        predictions = pd.read_csv('tests/data/test_sample_predictions.csv')
        self.gameweek_predictions = group_predictions_by_gameweek(predictions)
        self.elements = pd.read_csv('tests/data/test_elements.csv')


    def test_sum_gameweek_predictions(self):

        self.assertEqual(
            sum_gameweek_predictions([1, 2, 3], 1, self.gameweek_predictions), 23
        )
        self.assertEqual(
            sum_gameweek_predictions([1, 2, 3], 2, self.gameweek_predictions), 7
        )
        self.assertEqual(
            sum_gameweek_predictions([1, 2, 3], 3, self.gameweek_predictions), 10   
        )
        self.assertEqual(
            sum_gameweek_predictions([1, 2, 3], 4, self.gameweek_predictions), 16
        )
        self.assertEqual(
            sum_gameweek_predictions([1, 2, 3], 5, self.gameweek_predictions), 2
        )

        self.assertEqual(
            sum_gameweek_predictions([2], 1, self.gameweek_predictions), 13
        )
        self.assertEqual(
            sum_gameweek_predictions([2], 2, self.gameweek_predictions), 2
        )
        self.assertEqual(
            sum_gameweek_predictions([2], 3, self.gameweek_predictions), 5
        )
        self.assertEqual(
            sum_gameweek_predictions([2], 4, self.gameweek_predictions), 0
        )
        self.assertEqual(
            sum_gameweek_predictions([2], 5, self.gameweek_predictions), 0
        )

        self.assertEqual(
            sum_gameweek_predictions([1, 2], 4, self.gameweek_predictions), 8
        )
        self.assertEqual(
            sum_gameweek_predictions([1, 2], 5, self.gameweek_predictions), 1
        )

        self.assertEqual(
            sum_gameweek_predictions([1, 2, 3], 2, self.gameweek_predictions, 2), 14
        )
        self.assertEqual(
            sum_gameweek_predictions([1, 2, 3], 2, self.gameweek_predictions, [0.5, 1.5, 2.5]), 7.5
        )
        self.assertEqual(
            sum_gameweek_predictions([1, 2, 3], 3, self.gameweek_predictions, [0.5, 1.5, 2.5]), 10
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