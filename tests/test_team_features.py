import unittest

import pandas as pd
import numpy as np

from features.teams.conversion import EstimatedGoalsPerShot
from features.teams.xgd import UnderstatXGD

from utilities import compare_arrays


class TestTeamFeatures(unittest.TestCase):

    def setUp(self):
        self.teams = pd.read_csv('tests/data/test_team_features.csv')

    def test_estimated_goals_per_shot(self):

        EstimatedGoalsPerShot().engineer_features(self.teams)

        expected = np.array([0.28571, 0, 0.5, 0.6, 1, 0.66666, 0.4, 0.5, 0, 0])
        obtained = self.teams['est_goals_per_shot'].values

        return self.assertTrue(compare_arrays(obtained, expected))
    
    def test_understat_xgd(self):
        
        UnderstatXGD().engineer_features(self.teams)

        expected = np.array([-2.51, 1.60999, 0.7, 1.07, 0.2, 4.16, 2.39, 2.3, -1.02, -0.87999])
        obtained = self.teams['xGD'].values

        return self.assertTrue(compare_arrays(expected, obtained))