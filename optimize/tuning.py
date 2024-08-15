import optuna
import pandas as pd
import numpy as np

from simulation import run_simulation
from optimize.parameters import set_parameter


def evaluate():
    results = []

    for season in ['2021-22', '2022-23', '2023-24']:
        total_points = run_simulation(season)
        results.append(total_points)
    
    return np.mean(results)


def objective(trial: optuna.Trial):

    squad_evaluation_round_factor = trial.suggest_float('squad_evaluation_round_factor', 0.0, 1.0)
    reserve_gkp_multiplier = trial.suggest_float('reserve_gkp_multiplier', 0.0, 1.0)
    reserve_out_multiplier = trial.suggest_float('reserve_out_multiplier', 0.0, 1.0)
    reserve_out_multiplier = reserve_out_multiplier ** np.arange(1, 4)
    future_gameweeks_evaluated = trial.suggest_int('future_gameweeks_evaluated', 1, 10)
    budget_importance = trial.suggest_float('budget_importance', 1e-7, 1.0, log=True)

    set_parameter('squad_evaluation_round_factor', squad_evaluation_round_factor)
    set_parameter('reserve_gkp_multiplier', reserve_gkp_multiplier)
    set_parameter('reserve_out_multiplier', reserve_out_multiplier)
    set_parameter('future_gameweeks_evaluated', future_gameweeks_evaluated)
    set_parameter('budget_importance', budget_importance)

    return evaluate()


def tune_optimization_parameters():
    study = optuna.create_study(
        direction='maximize',
        storage="sqlite:///db.sqlite3",
        study_name="optimization-parameters",
        load_if_exists=True
    )
    study.optimize(objective)
    return study.best_params