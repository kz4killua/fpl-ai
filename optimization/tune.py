import random

import numpy as np
import optuna

from simulation.simulate import simulate


def objective(trial: optuna.Trial) -> float:
    # Suggest hyperparameters for the optimization
    optimization_window_size = trial.suggest_int("optimization_window_size", 3, 6)
    round_decay = trial.suggest_float("round_decay", 0.5, 1.0)
    vice_captain_multiplier = trial.suggest_float("vice_captain_multiplier", 1.0, 1.5)
    reserve_gkp_multiplier = trial.suggest_float("reserve_gkp_multiplier", 0.0, 0.2)
    reserve_out_1_multiplier = trial.suggest_float("reserve_out_1_multiplier", 0.0, 0.5)
    reserve_out_2_multiplier = trial.suggest_float("reserve_out_2_multiplier", 0.0, 0.3)
    reserve_out_3_multiplier = trial.suggest_float("reserve_out_3_multiplier", 0.0, 0.1)
    free_transfer_value = trial.suggest_float("free_transfer_value", 0.0, 5.0)
    budget_value = trial.suggest_float("budget_value", 1e-6, 1e-3, log=True)
    transfer_cost_multiplier = trial.suggest_float("transfer_cost_multiplier", 0.5, 2.0)
    # Evaluate the parameter combination with the simulation
    parameters = {
        "optimization_window_size": optimization_window_size,
        "round_decay": round_decay,
        "vice_captain_multiplier": vice_captain_multiplier,
        "reserve_gkp_multiplier": reserve_gkp_multiplier,
        "reserve_out_1_multiplier": reserve_out_1_multiplier,
        "reserve_out_2_multiplier": reserve_out_2_multiplier,
        "reserve_out_3_multiplier": reserve_out_3_multiplier,
        "free_transfer_value": free_transfer_value,
        "budget_value": budget_value,
        "transfer_cost_multiplier": transfer_cost_multiplier,
    }
    return evaluate(trial, parameters)


def evaluate(trial: optuna.Trial, parameters: dict) -> float:
    results = []
    for season in ["2021-22", "2022-23", "2023-24"]:
        # Simulate the season and get results
        random.seed(trial.number)
        wildcard_gameweeks = [random.randint(3, 19), random.randint(20, 36)]
        points = simulate(season, wildcard_gameweeks, parameters)
        results.append(points)
    # Calculate the average points across all seasons
    return np.mean(results)


def tune():
    study = optuna.create_study(
        direction="maximize",
        storage="sqlite:///db.sqlite3",
        study_name="optimization_tuning",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=None,
    )
    study.optimize(objective)
    return study.best_params
