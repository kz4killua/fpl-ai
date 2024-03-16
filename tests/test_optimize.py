import unittest

import pandas as pd
import numpy as np

from optimize.utilities import suggest_squad_roles, calculate_points, evaluate_squad, GKP, MID, DEF, FWD
from predictions import group_predictions_by_gameweek


class TestOptimize(unittest.TestCase):


    def setUp(self):

        predictions = pd.read_csv('tests/data/test_sample_predictions.csv')
        self.gameweek_predictions = group_predictions_by_gameweek(predictions)
        self.squad = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
        self.positions = pd.Series({
            1: GKP, 2: GKP, 3: DEF, 4: DEF, 5: DEF, 6: DEF, 7: DEF,
            8: MID, 9: MID, 10: MID, 11: MID, 12: MID,
            13: FWD, 14: FWD, 15: FWD
        })


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
        

    # TODO
    def test_get_valid_transfers(self):
        ...

    # TODO
    def test_make_best_transfer(self):
        ...