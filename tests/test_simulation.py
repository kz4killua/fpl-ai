import unittest

import pandas as pd
from optimize.utilities import GKP, MID, DEF, FWD
from simulation.utilities import make_automatic_substitutions


class TestMakeAutomaticSubstitutions(unittest.TestCase):

    def test_make_automatic_substitutions(self):

        # Entry 2267416 - 2023-24 season
        roles = {
            'captain': 415, 'vice_captain': 352,
            'starting_gkp': 352, 'starting_xi': [352, 616, 206, 398, 31, 6, 501, 294, 303, 415, 297],
            'reserve_gkp': 597, 'reserve_out': [368, 209, 278]
        }
        positions = pd.Series({
            352: GKP, 597: GKP, 
            616: DEF, 206: DEF, 398: DEF, 368: DEF, 31: DEF,
            209: MID, 6: MID, 501: MID, 294: MID, 303: MID,
            415: FWD, 297: FWD, 278: FWD
        })
        
        test_cases = [

            # Gameweek 1 - 1 substitution
            {
                'minutes': pd.Series({
                    352: 90, 616: 11, 206: 75, 398: 90, 209: 67, 6: 90, 501: 90, 
                    294: 65, 303: 76, 415: 67, 297: 65, 597: 90, 368: 0, 31: 0, 278: 32
                }), 
                'expected': {
                    'captain': 415, 'vice_captain': 352,
                    'starting_gkp': 352, 'starting_xi': [352, 616, 206, 398, 209, 6, 501, 294, 303, 415, 297],
                    'reserve_gkp': 597, 'reserve_out': [368, 31, 278]
                }, 
            },

            # Gameweek 3 - 0 substitutions
            {
                'minutes': pd.Series({
                    352: 90, 616: 90, 206: 0, 398: 0, 209: 0, 6: 55, 501: 90, 
                    294: 32, 303: 32, 415: 71, 297: 57, 597: 90, 368: 0, 31: 34, 278: 0
                }),
                'expected': {
                   'captain': 415, 'vice_captain': 352,
                   'starting_gkp': 352, 'starting_xi': [352, 616, 206, 398, 31, 6, 501, 294, 303, 415, 297],
                   'reserve_gkp': 597, 'reserve_out': [368, 209, 278]
                }
            },

            # Gameweek 8 - 2 substitutions
            {
                'minutes': pd.Series({
                    352: 90, 616: 90, 206: 0, 398: 0, 209: 62, 6: 15, 
                    501: 90, 294: 0, 303: 90, 415: 85, 297: 0, 597: 90, 368: 22, 31: 74, 278: 0
                }),
                'expected': {
                   'captain': 415, 'vice_captain': 352,
                   'starting_gkp': 352, 'starting_xi': [352, 616, 368, 209, 31, 6, 501, 294, 303, 415, 297],
                   'reserve_gkp': 597, 'reserve_out': [206, 398, 278]
                }
            },
        
            # Gameweek 9 - 3 substitutions
            {
                'minutes': pd.Series({
                    352: 0, 616: 90, 206: 6, 398: 0, 209: 2, 6: 12, 501: 90, 294: 90, 
                    303: 80, 415: 20, 297: 0, 597: 90, 368: 74, 31: 45, 278: 0
                }),
                'expected': {
                   'captain': 415, 'vice_captain': 352,
                   'starting_gkp': 597, 'starting_xi': [597, 616, 206, 368, 31, 6, 501, 294, 303, 415, 209],
                   'reserve_gkp': 352, 'reserve_out': [398, 297, 278]
                }
            },
        ]

        for test_case in test_cases:
            substituted_roles = make_automatic_substitutions(roles, test_case['minutes'], positions)
            expected = test_case['expected']
            self.assertEqual(substituted_roles['captain'], expected['captain'])
            self.assertEqual(substituted_roles['vice_captain'], expected['vice_captain'])
            self.assertEqual(substituted_roles['reserve_gkp'], expected['reserve_gkp'])
            self.assertListEqual(substituted_roles['reserve_out'], expected['reserve_out'])
            self.assertListEqual(substituted_roles['starting_xi'], expected['starting_xi'])
