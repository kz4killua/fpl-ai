import unittest

import pandas as pd
import numpy as np

from predictions import group_predictions_by_gameweek, sum_gameweek_predictions


class TestSumGameweekPredictions(unittest.TestCase):

    def setUp(self):
        predictions = pd.read_csv('tests/data/test_sample_predictions.csv')
        self.gameweek_predictions = group_predictions_by_gameweek(predictions)


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