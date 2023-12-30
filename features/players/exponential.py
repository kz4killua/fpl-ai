from ..common.exponential import ExponentialAverages


class PlayerExponentialAverages(ExponentialAverages):

    ITER_COLUMN = 'code'

    HALFLIFES = {
        'assists': 120,
        'bonus': 120,
        'bps': 25,
        'clean_sheets': 80,
        'creativity': 30,
        'goals_conceded': 40,
        'goals_scored': 80,
        'ict_index': 25,
        'influence': 35,
        'minutes': 7,
        'saves': 15,
        'threat': 30,
        'total_points': 40,
        'starts': 10,
        'expected_goals': 45,
        'expected_assists': 55,
        'expected_goal_involvements': 35,
        'expected_goals_conceded': 25,
        'understat_shots': 30,
        'understat_xG': 45,
        'understat_xA': 75,
        'understat_xGi': 40,
        'understat_key_passes': 40,
        'understat_xGChain': 40,
        'understat_xGBuildup': 50,
    }

    DATE_COLUMN = 'kickoff_time'