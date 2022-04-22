import pickle
import json

import pandas as pd
import numpy as np

import api
import data
import optimize


MAX_UPPER_GAMEWEEKS = 5
ANNEALING_ITERATIONS = 60000


def preprocess(df):
    """Carry out pre-prediction processing."""
    return df

def make_predictions(players, elements):
    """Predicts the number of points given
    player statistics.""" 

    # Load the model.
    with open('model/model.pkl', 'rb') as f:
        model = pickle.load(f)
    # Load feature data.
    with open('model/features.json') as g:
        features = json.load(g)

    # Select features we are interested in.
    X = players[features]
    # Make predictions.
    predictions = model.predict(X)

    # Map player ids to predicted points
    predictions = pd.Series(data=predictions, index=players['id'])

    # Sum up predicted points for each player.
    predictions = predictions.groupby(level=0).sum()

    # Predict 0 for any missing players.
    missing = list(set(elements['id']) - set(players['id']))
    missing = pd.Series(data=np.zeros(len(missing)), index=missing)

    predictions = pd.concat([predictions, missing])

    return predictions

def get_gameweek_matchups(players, gw, fixtures, teams):
    """Returns a dataframe of players mapped to their
    next opponents in a given gameweek."""

    # Create a DataFrame to hold our records.
    gameweek_matchups = pd.DataFrame()

    # Select fixtures in the given gameweek.
    gameweek_fixtures = fixtures[fixtures['event'] == gw]

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

    # Map opponent stats.
    # stats = [
    #     'strength'
    # ]
    # for stat in stats:
    #     gameweek_matchups['opponent_' + stat] = gameweek_matchups['opponent_team'].map(teams.set_index('id')[stat])

    return gameweek_matchups

def suggest_best_squad(my_team, next_gameweek_predictions, upper_gameweek_predictions, number_of_upper_gameweeks, elements, t):

    # Unpack team information.
    my_team_picks = pd.DataFrame(my_team['picks'])
    initial_budget_remaining = my_team['transfers']['bank']
    initial_squad = set(my_team_picks['element'])
    selling_prices = my_team_picks[['element', 'selling_price']]
    # Gather transfer information.
    transfer_limit = my_team['transfers']['limit']
    transfers_made = my_team['transfers']['made']
    transfer_cost = my_team['transfers']['cost']
    if transfer_limit is None:
        free_transfers = float('inf')
    else:
        free_transfers = transfer_limit - transfers_made

    # Suggest a best squad.
    best_squad = optimize.simulated_annealing(
        initial_squad, selling_prices, transfer_cost, free_transfers, initial_budget_remaining,
        next_gameweek_predictions, upper_gameweek_predictions, number_of_upper_gameweeks, elements, t)

    return best_squad

def main():

    season = '2021-22'

    # Request login details and attempt login.
    email = input('email: ')
    password = input('password: ')
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

    # Gather data from API
    print('Updating player data...')
    data.update_players_data(season, elements, events)

    print('Getting local player information...')
    players = data.get_all_current_data(season, elements, teams, next_gameweek)
    players = preprocess(players)

    print('Getting next gameweek matchups...')
    next_gameweek_matchups = get_gameweek_matchups(players, next_gameweek, fixtures, teams)

    print('Making predictions...')
    next_gameweek_predictions = make_predictions(next_gameweek_matchups, elements)
    # Scale next gameweek's predictions by availability.
    availability = elements.set_index('id')['chance_of_playing_next_round'] / 100
    next_gameweek_predictions *= availability

    if next_gameweek == last_gameweek:
        upper_gameweek_matchups = None
        upper_gameweek_predictions = None
        number_of_upper_gameweeks = 0

    else:
        print('Getting upper gameweek matchups...')
        upper_gameweek_matchups = []
        for gameweek in range(next_gameweek + 1, min(next_gameweek + 1 + MAX_UPPER_GAMEWEEKS, last_gameweek + 1)):
            # Get all player matchups for the given gameweek.
            matchups = get_gameweek_matchups(players, gameweek, fixtures, teams)
            # Add to the list.
            upper_gameweek_matchups.append(matchups)
        # Count the number of upper gameweeks.
        number_of_upper_gameweeks = len(upper_gameweek_matchups)
        # Concatenate all.
        upper_gameweek_matchups = pd.concat(upper_gameweek_matchups)

        print('Making predictions...')
        # Make predictions for upper gameweeks.
        upper_gameweek_predictions = make_predictions(upper_gameweek_matchups, elements)

    # Save predictions to a CSV file.
    out = elements[['id', 'first_name', 'second_name', 'team', 'element_type']].copy()
    out['team'] = out['team'].map(teams.set_index('id')['name'])
    out['element_type'] = out['element_type'].replace({1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"})
    out = out.set_index('id')
    out['next_gameweek_predictions'] = next_gameweek_predictions
    out['upper_gameweek_predictions'] = upper_gameweek_predictions
    out = out.reset_index()
    out.to_csv("predictions.csv")

    print('Suggesting best squad...')
    best_squad = suggest_best_squad(
        my_team, next_gameweek_predictions, upper_gameweek_predictions, number_of_upper_gameweeks, elements, ANNEALING_ITERATIONS)
    print('Suggesting squad roles...')
    squad_roles = optimize.suggest_squad_roles(best_squad, elements, next_gameweek_predictions)

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