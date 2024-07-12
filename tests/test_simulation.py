import unittest

import pandas as pd
from optimize.utilities import GKP, MID, DEF, FWD
from simulation.utilities import make_automatic_substitutions, calculate_selling_price, get_selling_prices, update_purchase_prices, update_selling_prices
from simulation.loaders import load_simulation_true_results, load_simulation_players_and_teams, load_simulation_purchase_prices


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


class TestSellingAndPurchasePrices(unittest.TestCase):

    def test_calculate_selling_price(self):
        
        test_cases = [
            {'purchase_price': 55, 'current_cost': 55, 'expected': 55},
            {'purchase_price': 50, 'current_cost': 49, 'expected': 49},
            {'purchase_price': 55, 'current_cost': 59, 'expected': 57},
            {'purchase_price': 50, 'current_cost': 53, 'expected': 51},
        ]

        for test_case in test_cases:
            self.assertEqual(
                calculate_selling_price(
                    test_case['purchase_price'],
                    test_case['current_cost'],
                ),
                test_case['expected']
            )


    def test_get_selling_prices(self):

        test_cases = [
            {
                'players': [1, 2, 3, 4], 
                'purchase_prices': pd.Series({
                    1: 55, 2: 50, 3: 55, 4: 50
                }),
                'now_costs': pd.Series({
                    1: 55, 2: 49, 3: 59, 4: 53
                }),
                'expected': pd.Series({
                    1: 55, 2: 49, 3: 57, 4: 51
                })
            }
        ]

        for test_case in test_cases:
            self.assertListEqual(
                list(
                    get_selling_prices(
                        test_case['players'],
                        test_case['purchase_prices'],
                        test_case['now_costs']
                    ).values
                ),
                list(test_case['expected'].values)
            )


    def test_update_purchase_prices(self):
        
        test_cases = [
            {
                'purchase_prices': pd.Series({1: 55, 2: 50, 3: 55, 4: 50}),
                'now_costs': pd.Series({1: 55, 2: 49, 3: 59, 4: 53, 5: 52, 6: 63}),
                'old_squad': {1, 2, 3, 4},
                'new_squad': {1, 2, 3, 5},
                'expected': pd.Series({1: 55, 2: 50, 3: 55, 5: 52}),
            },
            {
                'purchase_prices': pd.Series({1: 55, 2: 50, 3: 55, 4: 50}),
                'now_costs': pd.Series({1: 55, 2: 49, 3: 59, 4: 53, 5: 52, 6: 63}),
                'old_squad': {1, 2, 3, 4},
                'new_squad': {1, 2, 5, 6},
                'expected': pd.Series({1: 55, 2: 50, 5: 52, 6: 63}),
            }
        ]

        for test_case in test_cases:

            result = update_purchase_prices(
                test_case['purchase_prices'], 
                test_case['now_costs'], 
                test_case['old_squad'],
                test_case['new_squad']
            )
            self.assertTrue((result == test_case['expected']).all())


    def test_update_selling_prices(self):

        test_cases = [
            {
                'selling_prices': pd.Series({1: 55, 2: 50, 3: 55, 4: 50}),
                'now_costs': pd.Series({1: 55, 2: 49, 3: 59, 4: 53, 5: 52, 6: 63}),
                'old_squad': {1, 2, 3, 4},
                'new_squad': {1, 2, 3, 5},
                'expected': pd.Series({1: 55, 2: 50, 3: 55, 5: 52}),
            },
            {
                'selling_prices': pd.Series({1: 55, 2: 50, 3: 55, 4: 50}),
                'now_costs': pd.Series({1: 55, 2: 49, 3: 59, 4: 53, 5: 52, 6: 63}),
                'old_squad': {1, 2, 3, 4},
                'new_squad': {1, 2, 5, 6},
                'expected': pd.Series({1: 55, 2: 50, 5: 52, 6: 63}),
            }
        ]

        for test_case in test_cases:

            result = update_selling_prices(
                test_case['selling_prices'], 
                test_case['now_costs'], 
                test_case['old_squad'],
                test_case['new_squad']
            )
            self.assertTrue((result == test_case['expected']).all())


class TestLoaders(unittest.TestCase):

    def test_load_simulation_true_results(self):
        true_results = load_simulation_true_results('2023-24', use_cache=False)
        self.assertEqual(true_results['total_points'].loc[353, 38], 15)
        self.assertEqual(true_results['total_points'].loc[353, 37], 11)
        self.assertEqual(true_results['minutes'].loc[353, 37], 171)


    def test_load_simulation_purchase_prices(self):
        purchase_prices = load_simulation_purchase_prices('2023-24', {415, 6}, 1)
        self.assertEqual(purchase_prices.loc[415], 75)
        self.assertEqual(purchase_prices.loc[6], 75)
        
        purchase_prices = load_simulation_purchase_prices('2023-24', {415, 6}, 35)
        self.assertEqual(purchase_prices.loc[415], 82)
        self.assertEqual(purchase_prices.loc[6], 74)

    
    def test_load_simulation_players_and_teams(self):

        local_players, local_teams = load_simulation_players_and_teams('2017-18', 1)
        self.assertSequenceEqual(local_players['season'].max(), '2016-17')
        self.assertSequenceEqual(local_teams['fpl_season'].max(), '2016-17')
        self.assertEqual(local_players[local_players['season'] == '2016-17']['round'].max(), 38)
        self.assertEqual(len(local_teams[local_teams['fpl_season'] == '2017-18']), 0)
        self.assertEqual(len(local_teams[local_teams['fpl_season'] == '2017-18']), 0)
        self.assertEqual(len(local_teams[local_teams['fpl_season'] == '2016-17']), 38 * 20)

        local_players, local_teams = load_simulation_players_and_teams('2017-18', 10)
        self.assertSequenceEqual(local_players['season'].max(), '2017-18')
        self.assertSequenceEqual(local_teams['fpl_season'].max(), '2017-18')
        self.assertEqual(local_players[local_players['season'] == '2016-17']['round'].max(), 38)
        self.assertEqual(local_players[local_players['season'] == '2017-18']['round'].max(), 9)
        self.assertEqual(len(local_teams[local_teams['fpl_season'] == '2017-18']), 9 * 20)
        self.assertEqual(len(local_teams[local_teams['fpl_season'] == '2016-17']), 38 * 20)