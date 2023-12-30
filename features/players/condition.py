from ..common.condition import ConditionExponentialAverages


class PlayerConditionExponentialAverages(ConditionExponentialAverages):

    ITER_COLUMN = 'code'

    HALFLIFES = {
        'total_points': 140,
        'clean_sheets': 140,
    }

    DATE_COLUMN = 'kickoff_time'

    CONDITION_COLUMN = 'was_home'