from ..base import FeatureEngineeringStep


class EstimatedGoalsPerShot(FeatureEngineeringStep):
    """Compute an estimate of conversion rate."""

    def engineer_features(self, teams):
        
        conversion_rate = (
            teams['scored'] / (teams['scored'] + teams['missed'])
        )
        
        conversion_rate = conversion_rate.fillna(0)

        teams['est_goals_per_shot'] = conversion_rate

        return None