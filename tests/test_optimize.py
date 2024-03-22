import unittest

import pandas as pd
import numpy as np

from optimize.utilities import suggest_squad_roles, calculate_points, get_valid_transfers, evaluate_squad, make_best_transfer
from predictions import group_predictions_by_gameweek


class TestOptimize(unittest.TestCase):


    def setUp(self):

        predictions = pd.read_csv('tests/data/test_sample_predictions.csv')
        self.gameweek_predictions = group_predictions_by_gameweek(predictions)
        self.squad = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
        self.elements = pd.read_csv('tests/data/test_elements.csv').set_index('id', drop=False)
        self.elements['chance_of_playing_next_round'].fillna(100, inplace=True)
        self.selling_prices = pd.read_csv('tests/data/test_selling_prices.csv', index_col=['id'])['selling_price']
        self.positions = self.elements['element_type']


    def test_suggest_squad_roles(self):

        roles = suggest_squad_roles(self.squad, 1, self.positions, self.gameweek_predictions)
        self.assertEqual(roles['captain'], 2)
        self.assertEqual(roles['vice_captain'], 11)
        self.assertEqual(roles['reserve_gkp'], 1)
        self.assertSetEqual(set(roles['starting_xi']), {2, 3, 5, 11, 13, 4, 12, 6, 9, 8, 14})
        self.assertSetEqual(set(roles['reserve_out']), {7, 10, 15})

        roles = suggest_squad_roles(self.squad, 4, self.positions, self.gameweek_predictions)
        self.assertEqual(roles['captain'], 15)
        self.assertEqual(roles['vice_captain'], 4)
        self.assertEqual(roles['reserve_gkp'], 2)
        self.assertTrue(set(roles['starting_xi']).issuperset({1, 3, 4, 6, 8, 12, 13, 14, 15}))
        self.assertTrue(set(roles['reserve_out']).issubset({5, 7, 9, 10, 11}))


    def test_calculate_points(self):

        gameweek = 1
        roles = {
            'captain': 2, 'vice_captain': 11, 
            'starting_xi': [2, 5, 4, 3, 6, 11, 12, 9, 8, 13, 14], 
            'reserve_out': [7, 10, 15], 'reserve_gkp': 1
        }
        score = calculate_points(
            roles, self.gameweek_predictions, gameweek, 
            captain_multiplier=2,
            starting_xi_multiplier=1,
            reserve_gkp_multiplier=0.1,
            reserve_out_multiplier=np.array([0.3, 0.2, 0.1])
        )
        self.assertEqual(score, 90.5)

        gameweek = 4
        roles = {
            'captain': 15, 'vice_captain': 4, 
            'starting_xi': [1, 4, 3, 6, 5, 7, 8, 12, 15, 14, 13], 
            'reserve_out': [9, 10, 11], 'reserve_gkp': 2
        }
        score = calculate_points(
            roles, self.gameweek_predictions, gameweek, 
            captain_multiplier=2,
            starting_xi_multiplier=1,
            reserve_gkp_multiplier=0.1,
            reserve_out_multiplier=np.array([0.3, 0.2, 0.1])
        )
        self.assertEqual(score, 74)


    def test_evaluate_squad(self):
        
        gameweeks = [1, 2, 3]
        score = evaluate_squad(
            self.squad, self.positions, gameweeks, self.gameweek_predictions,
            squad_evaluation_round_factor=0.5,
            captain_multiplier=2,
            starting_xi_multiplier=1,
            reserve_gkp_multiplier=0.1,
            reserve_out_multiplier=np.array([0.3, 0.2, 0.1])
        )
        self.assertAlmostEqual(
            score, 76.84285714285714, 5
        )     


    def test_get_valid_transfers(self):

        test_cases = [
            # team restrictions
            {"player_out": 1, "budget": 100, "expected": {1, 16, 17}},
            {"player_out": 9, "budget": 100, "expected": {9, 23, 24, 25, 26, 27}},
            {"player_out": 10, "budget": 100, "expected": {10, 24, 25, 26, 27}},
            # budget restrictions
            {"player_out": 11, "budget": 100, "expected": {11, 24, 25, 26, 27}},
            {"player_out": 11, "budget": 0, "expected": {11, 27}},
            {"player_out": 12, "budget": 0, "expected": {12}},
        ]

        for test_case in test_cases:
            result = get_valid_transfers(self.squad, test_case['player_out'], self.elements, self.selling_prices, test_case['budget'])
            self.assertSetEqual(result, test_case['expected'])


    def test_make_best_transfer(self):

        test_cases = [
            {"gameweeks": [1], "budget": 0, "expected": {1, 2, 3, 4, 5, 6, 18, 8, 9, 10, 11, 12, 13, 14, 15}},
        ]

        for test_case in test_cases:
            squad = make_best_transfer(self.squad, test_case['gameweeks'], test_case['budget'], self.elements, self.selling_prices, self.gameweek_predictions)
            self.assertSetEqual(squad, test_case['expected'])