import requests
import json

import pandas as pd


def get_general_data():
    """Fetches and returns data from the 
    Fantasy Premier League API."""
    url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
    # Fetch data from the url.
    data = requests.get(url)
    # Convert to a JSON object.
    data = data.json()
    return data

def get_fixtures():
    url = 'https://fantasy.premierleague.com/api/fixtures/'
    # Fetch data from the url.
    data = requests.get(url)
    # Convert to a JSON object.
    data = data.json()
    return data

def get_live_data(event):
    """Returns players' information in a specific gameweek."""
    url = f"https://fantasy.premierleague.com/api/event/{event}/live/"
    return requests.get(url).json()

def get_player_data(element):
    """Returns a player's detailed information."""
    url = f"https://fantasy.premierleague.com/api/element-summary/{element}/"
    return requests.get(url).json()

def get_my_team_data(email, password):
    """Returns team information."""
    # Login.
    session = authenticate(email, password)
    # Get manager id.
    manager_data = session.get('https://fantasy.premierleague.com/api/me/')
    manager_id = manager_data.json()['player']['entry']
    # Get team data.
    team_data = session.get(f'https://fantasy.premierleague.com/api/my-team/{manager_id}/')
    # JSON-ize
    team_data = team_data.json()

    return team_data

def authenticate(email, password):
    """Returns an authenticated session."""
    # Create a session.
    session = requests.Session()

    url = 'https://users.premierleague.com/accounts/login/'
    headers = {'User-Agent': 'Dalvik/2.1.0 (Linux; U; Android 5.1; PRO 5 Build/LMY47D)'}
    payload = {
        'password': password,
        'login': email,
        'redirect_uri': 'https://fantasy.premierleague.com/a/login',
        'app': 'plfpl-web'
    }
    # Authenticate.
    session.post(url, data=payload, headers=headers)
    return session

def update_team(my_team, new_squad, squad_roles, elements, next_gameweek, email, password):
    """Logs into FPL and updates a team."""

    my_team_picks = pd.DataFrame(my_team['picks'])
    # Selling prices must be accessed by 'element' as id.
    selling_prices = my_team_picks.set_index('element')['selling_price']
    old_squad = set(my_team_picks['element'])

    # Check transfers in and out.
    transfers_in = list(new_squad - old_squad)
    transfers_out = list(old_squad - new_squad)

    # Sanity check.
    assert len(squad_roles['starting_xi']) == 11
    assert len(squad_roles['reserve_out']) == 3
    assert set(squad_roles['starting_xi']).isdisjoint(set(squad_roles['reserve_out']))
    assert squad_roles['reserve_gkp'] not in squad_roles['starting_xi']
    assert squad_roles['reserve_gkp'] not in squad_roles['reserve_out']

    # Login.
    session = authenticate(email, password)
    # Post transfers.
    post_transfers(session, transfers_in, transfers_out, elements, selling_prices, next_gameweek)
    # Post team sheet.
    post_squad_roles(session, squad_roles)

def post_transfers(session, transfers_in, transfers_out, elements, selling_prices, next_gameweek):
    """This function carries out transfers."""

    transfers = []

    # Sort transfers by position.
    transfers_in = tuple(elements[elements['id'].isin(transfers_in)].sort_values('element_type')['id'])
    transfers_out = tuple(elements[elements['id'].isin(transfers_out)].sort_values('element_type')['id'])

    # Build the transfers.
    for element_in, element_out in zip(transfers_in, transfers_out):
        # Retrieve cost and sale price.
        purchase_price = elements.set_index('id').loc[element_in, 'now_cost']
        selling_price = selling_prices.loc[element_out] # make sure index is element
        # Add to the transfer payload.
        transfer_details = {
            'element_in': int(element_in), 'element_out': int(element_out), 
            'purchase_price': int(purchase_price), 'selling_price': int(selling_price)
            }
        transfers.append(transfer_details)

    # This is the url we will be posting to.
    url = 'https://fantasy.premierleague.com/api/transfers/'

    # We will need some headers for the request.
    headers = {
        'content-type': 'application/json', 
        'origin': 'https://fantasy.premierleague.com', 
        'referer': 'https://fantasy.premierleague.com/transfers'}

    # Get manager id.
    manager_data = session.get('https://fantasy.premierleague.com/api/me/')
    manager_id = int(manager_data.json()['player']['entry'])

    # Create the transfer payload.
    payload = json.dumps({
        "transfers": transfers, "chip": None, "entry": manager_id, "event": int(next_gameweek)})

    # Post transfers.
    response = session.post(url=url, data=payload, headers=headers)

    return response

def post_squad_roles(session, squad_roles):
    """Logs into FPL and posts squad roles."""

    picks = []

    # Unpack the squad in order.
    players = [
        *squad_roles['starting_xi'], 
        squad_roles['reserve_gkp'], 
        *squad_roles['reserve_out']
    ]
    
    for position, element in enumerate(players, start=1):

        element_details = {
            'element': int(element), 
            'is_captain': bool(element == squad_roles['captain']),
            'is_vice_captain': bool(element == squad_roles['vice_captain']),
            'position': position
        }
        picks.append(element_details)

    # Get manager id.
    manager_data = session.get('https://fantasy.premierleague.com/api/me/')
    manager_id = int(manager_data.json()['player']['entry'])

    url = f'https://fantasy.premierleague.com/api/my-team/{manager_id}/'

    payload = json.dumps({"picks": picks, "chip": None})

    headers = {
        'content-type': 'application/json', 
        'origin': 'https://fantasy.premierleague.com', 
        'referer': 'https://fantasy.premierleague.com/my-team'}

    response = session.post(url=url, data=payload, headers=headers)

    return response


# Test posting when no transfers were made.