from ..base import FeatureEngineeringStep


class UnderstatXGD(FeatureEngineeringStep):
    """Compute expected goal difference using understat.com data."""

    def engineer_features(self, teams):

        teams['xGD'] = teams['xG'] - teams['xGA']

        return None