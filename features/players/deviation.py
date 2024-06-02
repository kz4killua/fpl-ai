from ..common.deviation import StandardDeviation


class PlayerStandardDeviation(StandardDeviation):

    ITER_COLUMN = 'code'

    WINDOWS = {
        'total_points': 10,
        'clean_sheets': 10,
    }

    def get_feature_name(self, column, window):
        return f'std_{column}_{window}'