from .pipeline import FeaturePipeline

from .players.categorical import PlayerOneHotEncode
from .players.condition import PlayerConditionExponentialAverages
from .players.deviation import PlayerStandardDeviation
from .players.exponential import PlayerExponentialAverages
from .players.xgi import UnderstatXGI
from .players.position import PositionAveragesAgainstOpponent
from .players.predictedposition import PredictedPlayerPosition

from .teams.xgd import UnderstatXGD
from .teams.conversion import EstimatedGoalsPerShot
from .teams.exponential import TeamExponentialAverages
from .teams.condition import TeamConditionExponentialAverages
from .teams.deviation import TeamStandardDeviation


class PlayerFeaturePipeline(FeaturePipeline):

    STEPS = [
        UnderstatXGI(),
        PredictedPlayerPosition(),
        PositionAveragesAgainstOpponent(),
        PlayerOneHotEncode(),
        PlayerExponentialAverages(),
        PlayerConditionExponentialAverages(),
        PlayerStandardDeviation(),
    ]


class TeamFeaturePipeline(FeaturePipeline):

    STEPS = [
        UnderstatXGD(),
        EstimatedGoalsPerShot(),
        TeamExponentialAverages(),
        TeamConditionExponentialAverages(),
        TeamStandardDeviation(),
    ]


def engineer_features(players, teams):

    # Engineer features for players
    players, player_features = PlayerFeaturePipeline().apply(players)

    # Engineer features for teams
    teams, team_features = TeamFeaturePipeline().apply(teams)

    # Merge players and teams
    merged = merge_players_and_teams(players, teams, team_features)

    # Identify columns to be used for prediction
    columns = [

        'round', 'was_home',

        *player_features,

        *[
            f"team_{column}" for column in team_features
        ],

        *[
            f"opponent_team_{column}" for column in team_features
        ],

        'total_points'

    ]

    return merged[columns]


def merge_players_and_teams(players, teams, team_features):
    """Add team engineered features to the players dataframe."""
    
    # Create re-indexed dataframes to make merging easier
    indexed_players_team = players.set_index(['season', 'team_code', 'fixture'])
    indexed_players_opponent = players.set_index(['season', 'opponent_team_code', 'fixture'])
    indexed_teams = teams.set_index(['fpl_season', 'fpl_code', 'fpl_fixture_id'])

    for column in team_features:

        # Add features for the player's team
        players[f"team_{column}"] = indexed_players_team.index.map(
            indexed_teams[column]
        )

        # Add features for the opponent team
        players[f"opponent_team_{column}"] = indexed_players_opponent.index.map(
            indexed_teams[column]
        )

    return players