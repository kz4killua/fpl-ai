from ..common.exponential import ExponentialAverages


class TeamExponentialAverages(ExponentialAverages):

    ITER_COLUMN = 'fpl_code'
    
    HALFLIFES = {
        'xG': 160,
        'xGA': 230,
        'deep': 170,
        'deep_allowed': 355,
        'scored': 320,
        'missed': 275,
        'xpts': 150,
        'wins': 355,
        'draws': 345, 
        'loses': 375,
        'pts': 310,
        'npxGD': 170,
        'ppda_att': 180,
        'ppda_def': 395,
        'ppda_allowed_att': 120,
        'ppda_allowed_def': 60,
        'est_goals_per_shot': 180,
    }

    DATE_COLUMN = 'date'