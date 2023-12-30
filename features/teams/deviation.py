from ..common.deviation import StandardDeviation


class TeamStandardDeviation(StandardDeviation):

    ITER_COLUMN = 'fpl_code'

    WINDOWS = {
        'xG': 35,
        'xGA': 35, 
        'deep': 35, 
        'deep_allowed': 35,
        'scored': 35,
        'missed': 35,
        'xpts': 35,
        'wins': 35,
        'draws': 35, 
        'loses': 35,
        'pts': 35,
        'npxGD': 35,
        'ppda_att': 35,
        'ppda_def': 35,
        'ppda_allowed_att': 35,
        'ppda_allowed_def': 35,
        'est_goals_per_shot': 35,
    }

    def get_feature_name(self, column, window):
        return f'average_std_{column}_{window}'