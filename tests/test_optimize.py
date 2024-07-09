import unittest

import pandas as pd
import numpy as np

from optimize.utilities import suggest_squad_roles, calculate_points, get_valid_transfers, evaluate_squad, make_best_transfer, calculate_budget, get_future_gameweeks
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

        gameweek = 1
        roles = suggest_squad_roles(self.squad, self.positions, self.gameweek_predictions.loc[:, gameweek])
        self.assertEqual(roles['captain'], 2)
        self.assertEqual(roles['vice_captain'], 11)
        self.assertEqual(roles['reserve_gkp'], 1)
        self.assertSetEqual(set(roles['starting_xi']), {2, 3, 5, 11, 13, 4, 12, 6, 9, 8, 14})
        self.assertSetEqual(set(roles['reserve_out']), {7, 10, 15})

        gameweek = 4
        roles = suggest_squad_roles(self.squad, self.positions, self.gameweek_predictions.loc[:, gameweek])
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
            roles=roles,
            total_points=self.gameweek_predictions.loc[:, gameweek],
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
            roles=roles,
            total_points=self.gameweek_predictions.loc[:, gameweek],
            captain_multiplier=2,
            starting_xi_multiplier=1,
            reserve_gkp_multiplier=0.1,
            reserve_out_multiplier=np.array([0.3, 0.2, 0.1])
        )
        self.assertEqual(score, 74)


    def test_evaluate_squad(self):
        
        gameweeks = [1, 2, 3]
        positions = self.positions.to_dict()
        gameweek_predictions = {
            gameweek: (
                self.gameweek_predictions.loc[:, gameweek].to_dict()
            )
            for gameweek in gameweeks
        }

        score = evaluate_squad(
            self.squad, positions, gameweeks, gameweek_predictions,
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


    def test_calculate_budget(self):
        
        test_cases = [
            {
                'initial_squad': {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
                'final_squad': {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
                'selling_prices': self.selling_prices,
                'now_costs': self.elements['now_cost'],
                'initial_budget': 10,
                'expected': 10
            },

            {
                'initial_squad': {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
                'final_squad': {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 30},
                'selling_prices': self.selling_prices,
                'now_costs': self.elements['now_cost'],
                'initial_budget': 10,
                'expected': 22
            }, 

            {
                'initial_squad': {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
                'final_squad': {17, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 30},
                'selling_prices': self.selling_prices,
                'now_costs': self.elements['now_cost'],
                'initial_budget': 10,
                'expected': 26
            },
        ]

        for test_case in test_cases:
            self.assertEqual(
                calculate_budget(
                    test_case['initial_squad'],
                    test_case['final_squad'],
                    test_case['initial_budget'],
                    test_case['selling_prices'],
                    test_case['now_costs']
                ),
                test_case['expected']
            )


    def test_get_future_gameweeks(self):

        test_cases = [
            {
                'next_gameweek': 1, 
                'last_gameweek': 38, 
                'future_gameweeks_evaluated': 5, 
                'expected': [1, 2, 3, 4, 5],
                'wildcard_gameweeks': []
            },
            {
                'next_gameweek': 36, 
                'last_gameweek': 38, 
                'future_gameweeks_evaluated': 5, 
                'expected': [36, 37, 38],
                'wildcard_gameweeks': []
            },
            {
                'next_gameweek': 5, 
                'last_gameweek': 8, 
                'future_gameweeks_evaluated': 7, 
                'expected': [5, 6, 7, 8],
                'wildcard_gameweeks': []
            },
            {
                'next_gameweek': 5, 
                'last_gameweek': 38, 
                'future_gameweeks_evaluated': 3, 
                'expected': [5, 6, 7],
                'wildcard_gameweeks': []
            },
            {
                'next_gameweek': 1, 
                'last_gameweek': 38, 
                'future_gameweeks_evaluated': 5, 
                'expected': [1, 2, 3, 4, 5],
                'wildcard_gameweeks': [10, 26],
            },
            {
                'next_gameweek': 36, 
                'last_gameweek': 38, 
                'future_gameweeks_evaluated': 5, 
                'expected': [36, 37, 38],
                'wildcard_gameweeks': [10, 26], 
            },
            {
                'next_gameweek': 7, 
                'last_gameweek': 38, 
                'future_gameweeks_evaluated': 5, 
                'expected': [7, 8, 9],
                'wildcard_gameweeks': [10, 26], 
            },
            {
                'next_gameweek': 10, 
                'last_gameweek': 38, 
                'future_gameweeks_evaluated': 5, 
                'expected': [10, 11, 12, 13, 14],
                'wildcard_gameweeks': [10, 26], 
            },
            {
                'next_gameweek': 10, 
                'last_gameweek': 38, 
                'future_gameweeks_evaluated': 5, 
                'expected': [10, 11, 12],
                'wildcard_gameweeks': [10, 13], 
            },
            {
                'next_gameweek': 38, 
                'last_gameweek': 38, 
                'future_gameweeks_evaluated': 5, 
                'expected': [38],
                'wildcard_gameweeks': [38], 
            },
            {
                'next_gameweek': 15, 
                'last_gameweek': 38, 
                'future_gameweeks_evaluated': 10, 
                'expected': [15, 16, 17, 18],
                'wildcard_gameweeks': [14, 19], 
            },
        ]

        for test_case in test_cases:
            result = get_future_gameweeks(
                next_gameweek=test_case['next_gameweek'], 
                last_gameweek=test_case['last_gameweek'],
                wildcard_gameweeks=test_case['wildcard_gameweeks'],
                future_gameweeks_evaluated=test_case['future_gameweeks_evaluated']
            )
            self.assertListEqual(result, test_case['expected'])