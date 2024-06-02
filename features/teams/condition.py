from ..common.condition import ConditionExponentialAverages


class TeamConditionExponentialAverages(ConditionExponentialAverages):

    ITER_COLUMN = 'fpl_code'

    HALFLIFES = {
        'xG': 205,
        'xGA': 345, 
        'deep': 205, 
        'deep_allowed': 395,
        'scored': 395,
        'missed': 395,
        'xpts': 190,
        'wins': 395,
        'draws': 300, 
        'loses': 395,
        'pts': 395,
        'npxGD': 240,
        'ppda_att': 395,
        'ppda_def': 395,
        'ppda_allowed_att': 145,
        'ppda_allowed_def': 75,
        'est_goals_per_shot': 200,
    }

    DATE_COLUMN = 'date'

    CONDITION_COLUMN = 'h_a'