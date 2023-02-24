import os
import pickle
import json

import pandas as pd
import numpy as np

import api
import data
import optimize

# Load the model and feature data
with open('model/model.pkl', 'rb') as f:
    MODEL = pickle.load(f)
with open('model/features.json') as g:
    FEATURES = json.load(g)


def preprocess(players):
    """Carries out pre-prediction processing."""
    return players


def make_predictions(players, elements):
    """Predicts the number of points given
    player statistics."""

    # Select features we are interested in.
    X = players[FEATURES]
    # Make predictions.
    predictions = MODEL.predict(X)

    # Map player ids to predicted points
    predictions = pd.Series(data=predictions, index=players['id'])

    # Sum up predicted points for each player.
    predictions = predictions.groupby(level=0).sum()

    # Predict 0 for any missing players.
    missing = list(set(elements['id']) - set(players['id']))
    missing = pd.Series(data=np.zeros(len(missing)), index=missing)

    predictions = pd.concat([predictions, missing])

    return predictions


def get_gameweek_matchups(players, next_gameweek, fixtures):
    """Returns a dataframe of players mapped to each of their
    opponents in a given gameweek."""

    # Create a DataFrame to hold our records.
    gameweek_matchups = pd.DataFrame()

    # Select fixtures in the given gameweek.
    gameweek_fixtures = fixtures[fixtures['event'] == next_gameweek]

    for index, row in gameweek_fixtures.iterrows():

        # Get home and away teams for each fixture.
        home = players[players['team'] == row['team_h']].copy()
        away = players[players['team'] == row['team_a']].copy()

        # Set opponent teams.
        home['opponent_team'] = row['team_a']
        away['opponent_team'] = row['team_h']

        # Set home and away information.
        home['is_home'] = 1
        away['is_home'] = 0

        # Add fixture difficulty ratings.
        home['team_fixture_difficulty'] = row['team_h_difficulty']
        away['team_fixture_difficulty'] = row['team_a_difficulty']
        home['opponent_fixture_difficulty'] = row['team_a_difficulty']
        away['opponent_fixture_difficulty'] = row['team_h_difficulty']

        # Add our data to a new dataframe.
        gameweek_matchups = pd.concat([gameweek_matchups, home, away])

    return gameweek_matchups


def suggest_best_squad(my_team, next_gameweek_predictions, upper_gameweek_predictions, elements):
    """Optimizes a team to get best transfers."""

    # Unpack team information.
    picks = pd.DataFrame(my_team['picks'])
    initial_budget_remaining = my_team['transfers']['bank']
    initial_squad = set(picks['element'])
    selling_prices = picks.set_index('element')['selling_price']
    # Gather transfer information.
    transfer_limit = my_team['transfers']['limit']
    transfers_made = my_team['transfers']['made']
    if transfer_limit is None:
        free_transfers = float('inf')
    else:
        free_transfers = transfer_limit - transfers_made

    # Suggest a best squad.
    best_squad = optimize.best_transfer(
        initial_squad, initial_budget_remaining, elements, 
        selling_prices, next_gameweek_predictions, upper_gameweek_predictions,
        free_transfers
    )

    return best_squad


def save_predictions(elements, teams, next_gameweek_predictions, upper_gameweek_predictions, next_gameweek):
    """Saves predictions to a CSV file."""
    predictions = elements[['id', 'first_name', 'second_name', 'team', 'element_type']].copy()
    predictions['team'] = predictions['team'].map(teams.set_index('id')['name'])
    predictions['element_type'].replace({1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}, inplace=True)
    predictions = predictions.set_index('id', drop=False)
    predictions[f'gw_{next_gameweek}'] = next_gameweek_predictions
    for gameweek, gameweek_predictions in enumerate(
        upper_gameweek_predictions, start=next_gameweek+1):
        predictions[f'gw_{gameweek}'] = gameweek_predictions
    predictions.to_csv("predictions.csv", index=False)


def get_gameweek_predictions(players, gameweek, fixtures, elements):
    """Predicts points for a given gameweek."""
    matchups = get_gameweek_matchups(players, gameweek, fixtures)
    predictions = make_predictions(matchups, elements)
    return predictions


def get_next_gameweek_predictions(players, next_gameweek, fixtures, elements):
    """Predicts points for the next gameweek."""
    predictions = get_gameweek_predictions(players, next_gameweek, fixtures, elements)
    # Scale by availability.
    availability = elements.set_index('id')['chance_of_playing_next_round'] / 100
    predictions *= availability
    return predictions


def list_upper_gameweeks(next_gameweek, last_gameweek):
    """Returns the ids of upper gameweeks."""
    return range(
        next_gameweek + 1, 
        min(
            next_gameweek + optimize.MAX_UPPER_GAMEWEEKS, 
            last_gameweek
            ) + 1
        )


def get_upper_gameweek_predictions(players, next_gameweek, last_gameweek, fixtures, elements):
    """Predicts points for upper gameweeks."""
    upper_gameweek_predictions = [
        get_gameweek_predictions(players, gameweek, fixtures, elements)
        for gameweek in list_upper_gameweeks(next_gameweek, last_gameweek)
    ]
    # Scale by availability.
    availability = elements.set_index('id')['status'].replace(
        {'a': 1, 'd': 1, 'i': 0, 'u': 0, 'n': 1, 's': 1}).astype('int')
    upper_gameweek_predictions = [
        (predictions * availability)
        for predictions in upper_gameweek_predictions
    ]
    return upper_gameweek_predictions


def main():

    # Get login details
    if "FPL_EMAIL" in os.environ and "FPL_PASSWORD" in os.environ:
        email = os.environ.get("FPL_EMAIL")
        password = os.environ.get("FPL_PASSWORD")
    else:
        email = input('FPL email: ')
        password = input('FPL password: ')

    # Get data from the API
    print('Getting manager information...')
    my_team = api.get_my_team_data(email, password)
    print('Getting general information...')
    general_data = api.get_general_data()
    print('Getting fixtures...')
    fixtures = api.get_fixtures()

    # Unpack data
    teams = pd.DataFrame(general_data['teams'])
    events = pd.DataFrame(general_data['events'])
    elements = pd.DataFrame(general_data['elements'])
    elements['chance_of_playing_next_round'].fillna(100, inplace=True)
    fixtures = pd.DataFrame(fixtures)
    next_gameweek = events[events['is_next'] == True].iloc[0]['id']
    last_gameweek = events['id'].max()
    
    season = data.get_current_season(fixtures)

    print('Updating player data...')
    data.update_players_data(season, elements, events)

    print('Getting local player information...')
    players = data.get_current_data(season, elements, next_gameweek)
    players = preprocess(players)

    print("Making predictions...")
    next_gameweek_predictions = get_next_gameweek_predictions(players, next_gameweek, fixtures, elements)
    upper_gameweek_predictions = get_upper_gameweek_predictions(players, next_gameweek, last_gameweek, fixtures, elements)

    # Save predictions to a CSV file
    save_predictions(
        elements, teams, next_gameweek_predictions, 
        upper_gameweek_predictions, next_gameweek
    )

    print('Suggesting best squad...')
    best_squad = suggest_best_squad(
        my_team, next_gameweek_predictions, upper_gameweek_predictions, elements)
    squad_roles = optimize.suggest_squad_roles(best_squad, elements.set_index('id')['element_type'], next_gameweek_predictions)

    print('Updating team...')
    api.update_team(my_team, best_squad, squad_roles, elements, next_gameweek, email, password)

    print('Finished!')



if __name__ == '__main__':    
    try:
        main()
    except:
        print("""
        xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        An error occurred. Please try running again
        with a stable internet connection.
        xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        """)
    else:
        print("""
        -----------------------------------------------
        SUCCESS. TEAM UPDATED.
        -----------------------------------------------
        """)
    finally:
        input('Press Enter to exit.')