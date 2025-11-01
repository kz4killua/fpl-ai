from collections import defaultdict

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.stats import poisson

from loaders.utils import force_dataframe


def compute_toa_features(
    matches: pl.LazyFrame | pl.DataFrame, bookmaker_weights: dict
) -> pl.LazyFrame:
    matches = force_dataframe(matches)

    predictions = []
    for match in matches.to_dicts():
        bookmakers = match["toa_bookmakers"]
        if not bookmakers:
            continue

        home_team = match["team_h_toa_name"]
        away_team = match["team_a_toa_name"]

        # Calculate implied probabilities for each market
        markets = defaultdict(list)
        for bookmaker in bookmakers:
            if bookmaker["key"] not in bookmaker_weights:
                continue

            for market in bookmaker["markets"]:
                if market["key"] == "h2h":
                    data = extract_h2h_probs(market, home_team, away_team)
                elif market["key"] == "spreads":
                    data = extract_spreads_probs(market, home_team, away_team)
                elif market["key"] == "totals":
                    data = extract_totals_probs(market)
                else:
                    continue

                markets[market["key"]].append(
                    {
                        "bookmaker": bookmaker["key"],
                        "type": market["key"],
                        "outcomes": data,
                    }
                )

        if not markets:
            continue

        # Aggregate market probabilities across bookmakers
        aggregated_markets = dict()
        for key in markets:
            if key == "h2h":
                aggregated_markets[key] = aggregate_h2h_probs(
                    markets[key], bookmaker_weights
                )

            elif key == "spreads":
                aggregated_markets[key] = aggregate_spreads_probs(
                    markets[key], bookmaker_weights
                )
            elif key == "totals":
                aggregated_markets[key] = aggregate_totals_probs(
                    markets[key], bookmaker_weights
                )
            else:
                continue

        # Predict expected scores using the aggregated probabilities
        target_probs = []
        for key, data in aggregated_markets.items():
            target_probs.append({"type": key, "data": data})

        result = estimate_goals(target_probs, max_goals=10)
        if result.success:
            predicted_home_goals, predicted_away_goals = result.x
        else:
            predicted_home_goals = predicted_away_goals = None

        # Store the predictions for goals and win probabilities
        row = {
            "season": match["season"],
            "fixture_id": match["fixture_id"],
            "team_h_toa_win_prob": None,
            "team_a_toa_win_prob": None,
            "team_h_toa_expected_goals": predicted_home_goals,
            "team_a_toa_expected_goals": predicted_away_goals,
        }

        for value in aggregated_markets.get("h2h", []):
            if value["side"] == "home":
                row["team_h_toa_win_prob"] = value["prob"]
            elif value["side"] == "away":
                row["team_a_toa_win_prob"] = value["prob"]

        predictions.append(row)

    toa_predictions = pl.DataFrame(predictions)

    # Merge predictions back into the original matches DataFrame
    matches = matches.join(
        toa_predictions.select(
            "season",
            "fixture_id",
            "team_h_toa_expected_goals",
            "team_a_toa_expected_goals",
            "team_h_toa_win_prob",
            "team_a_toa_win_prob",
        ),
        on=["season", "fixture_id"],
        how="left",
    )

    return matches.lazy()


def extract_h2h_probs(market: dict, home_team: str, away_team: str) -> dict:
    home_odds = away_odds = draw_odds = None
    for outcome in market["outcomes"]:
        if outcome["name"] == home_team:
            home_odds = outcome["price"]
        elif outcome["name"] == away_team:
            away_odds = outcome["price"]
        elif outcome["name"] == "Draw":
            draw_odds = outcome["price"]

    if home_odds is None or away_odds is None or draw_odds is None:
        raise ValueError("Missing odds for one of the outcomes in h2h market.")

    home_prob, away_prob, draw_prob = calculate_implied_probs(
        home_odds, away_odds, draw_odds
    )
    data = [
        {"side": "home", "prob": home_prob},
        {"side": "away", "prob": away_prob},
        {"side": "draw", "prob": draw_prob},
    ]
    return data


def extract_spreads_probs(market: dict, home_team: str, away_team: str) -> dict:
    probs = calculate_implied_probs(
        *[outcome["price"] for outcome in market["outcomes"]]
    )

    data = []
    for outcome, prob in zip(market["outcomes"], probs, strict=True):
        if outcome["name"] == home_team:
            side = "home"
        elif outcome["name"] == away_team:
            side = "away"
        else:
            raise ValueError("Unexpected outcome name in spreads market.")

        data.append({"side": side, "point": outcome["point"], "prob": prob})

    return data


def extract_totals_probs(market: dict) -> dict:
    probs = calculate_implied_probs(
        *[outcome["price"] for outcome in market["outcomes"]]
    )

    data = []
    for outcome, prob in zip(market["outcomes"], probs, strict=True):
        data.append(
            {
                "name": outcome["name"],
                "point": outcome["point"],
                "prob": prob,
            }
        )

    return data


def aggregate_h2h_probs(bookmakers: list[dict], bookmaker_weights: dict) -> list[dict]:
    groups = defaultdict(list)
    for bookmaker in bookmakers:
        for outcome in bookmaker["outcomes"]:
            groups[outcome["side"]].append(
                {"bookmaker": bookmaker["bookmaker"], "prob": outcome["prob"]}
            )

    aggregated = []
    for side, outcomes in groups.items():
        aggregated.append(
            {"side": side, "prob": weight_probs(outcomes, bookmaker_weights)}
        )

    return aggregated


def aggregate_spreads_probs(
    bookmakers: list[dict], bookmaker_weights: dict | defaultdict
) -> list[dict]:
    groups = defaultdict(list)
    for bookmaker in bookmakers:
        for outcome in bookmaker["outcomes"]:
            key = (outcome["side"], outcome["point"])
            groups[key].append(
                {"bookmaker": bookmaker["bookmaker"], "prob": outcome["prob"]}
            )

    aggregated = []
    for (side, point), outcomes in groups.items():
        aggregated.append(
            {
                "side": side,
                "point": point,
                "prob": weight_probs(outcomes, bookmaker_weights),
            }
        )

    return aggregated


def aggregate_totals_probs(
    bookmakers: list[dict], bookmaker_weights: dict | defaultdict
) -> list[dict]:
    groups = defaultdict(list)
    for bookmaker in bookmakers:
        for outcome in bookmaker["outcomes"]:
            key = (outcome["name"], outcome["point"])
            groups[key].append(
                {"bookmaker": bookmaker["bookmaker"], "prob": outcome["prob"]}
            )

    aggregated = []
    for (name, point), outcomes in groups.items():
        aggregated.append(
            {
                "name": name,
                "point": point,
                "prob": weight_probs(outcomes, bookmaker_weights),
            }
        )

    return aggregated


def weight_probs(outcomes: list[dict], bookmaker_weights: dict) -> float:
    """Compute a weighted average of probabilities from different bookmakers."""

    bookmakers, probs, weights = [], [], []
    for outcome in outcomes:
        bookmakers.append(outcome["bookmaker"])
        probs.append(outcome["prob"])
        weights.append(bookmaker_weights[outcome["bookmaker"]])

    total_weight = sum(weights)
    if total_weight == 0:
        raise ZeroDivisionError(
            "Sum of bookmaker weights is zero. Cannot compute weighted average."
        )

    weights = [w / total_weight for w in weights]
    return sum([v * p for v, p in zip(probs, weights, strict=True)])


def compute_h2h_error(prob_matrix: np.ndarray, target_prob: list):
    # Sum up the observed probabilities for home win, away win, and draw events
    observed = [0 for _ in target_prob["data"]]

    for h_goals in range(prob_matrix.shape[0]):
        for a_goals in range(prob_matrix.shape[1]):
            for i, event in enumerate(target_prob["data"]):
                if event["side"] == "home" and h_goals > a_goals:
                    observed[i] += prob_matrix[h_goals, a_goals]
                if event["side"] == "away" and a_goals > h_goals:
                    observed[i] += prob_matrix[h_goals, a_goals]
                if event["side"] == "draw" and h_goals == a_goals:
                    observed[i] += prob_matrix[h_goals, a_goals]

    # Compute the sum of squared errors between observed and target probabilities
    error = sum(
        (observed[i] - event["prob"]) ** 2
        for i, event in enumerate(target_prob["data"])
    )

    return error


def compute_spreads_error(prob_matrix: np.ndarray, target_prob: list):
    # Sum up the observed probabilities for all target spreads
    observed = [0 for _ in target_prob["data"]]

    for h_goals in range(prob_matrix.shape[0]):
        for a_goals in range(prob_matrix.shape[1]):
            for i, spread in enumerate(target_prob["data"]):
                if spread["side"] == "home" and (h_goals + spread["point"]) > a_goals:
                    observed[i] += prob_matrix[h_goals, a_goals]
                if spread["side"] == "away" and (a_goals + spread["point"]) > h_goals:
                    observed[i] += prob_matrix[h_goals, a_goals]

    # Compute the squared error
    error = sum(
        (observed[i] - spread["prob"]) ** 2
        for i, spread in enumerate(target_prob["data"])
    )

    return error


def compute_totals_error(prob_matrix: np.ndarray, target_prob: list):
    # Sum up the observed probabilities for all over/under lines
    observed = [0 for _ in target_prob["data"]]

    for h_goals in range(prob_matrix.shape[0]):
        for a_goals in range(prob_matrix.shape[1]):
            for i, line in enumerate(target_prob["data"]):
                if line["name"] == "Over" and (h_goals + a_goals) > line["point"]:
                    observed[i] += prob_matrix[h_goals, a_goals]
                if line["name"] == "Under" and (h_goals + a_goals) < line["point"]:
                    observed[i] += prob_matrix[h_goals, a_goals]

    # Calculate squared error
    error = sum(
        (observed[i] - line["prob"]) ** 2 for i, line in enumerate(target_prob["data"])
    )

    return error


def objective(params: list, target_probs: dict, max_goals: int):
    lambda_h, lambda_a = params

    # Penalize non-positive parameters
    if lambda_h <= 0 or lambda_a <= 0:
        return float("inf")

    # Calculate the probabilities of each scoreline (assuming a Poisson distribution)
    goals_range = np.arange(max_goals + 1)
    prob_h = poisson.pmf(goals_range, lambda_h)
    prob_a = poisson.pmf(goals_range, lambda_a)
    prob_matrix = np.outer(prob_h, prob_a)

    # Compute the total error across all target probabilities
    errors = []
    for target_prob in target_probs:
        if target_prob["type"] == "h2h":
            error = compute_h2h_error(prob_matrix, target_prob)
        elif target_prob["type"] == "spreads":
            error = compute_spreads_error(prob_matrix, target_prob)
        elif target_prob["type"] == "totals":
            error = compute_totals_error(prob_matrix, target_prob)
        else:
            raise ValueError(f"Unknown target probability type: {target_prob['type']}")
        errors.append(error)

    return sum(errors)


def estimate_goals(target_probs: dict, max_goals: int) -> dict:
    """Estimate expected goals for home and away teams based on target probabilities."""
    initial_guess = [1.5, 1.2]
    bounds = [(1e-6, max_goals), (1e-6, max_goals)]
    result = minimize(
        objective,
        initial_guess,
        args=(
            target_probs,
            max_goals,
        ),
        method="L-BFGS-B",
        bounds=bounds,
    )
    return result


def calculate_implied_probs(*values: list[float]) -> list[float]:
    """Convert bookmaker odds to implied probabilities, adjusted for overround."""
    implied = [1 / v for v in values]
    overround = sum(implied)
    normalized = [x / overround for x in implied]
    return normalized
