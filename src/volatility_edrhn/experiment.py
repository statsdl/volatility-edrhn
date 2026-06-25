from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable

import numpy as np

from .data import chronological_split, generate_synthetic_volatility, make_supervised_frame, minmax_scale_splits
from .metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
from .model import EDRHNRegressor
from .tuning import tune_edrhn


@dataclass(frozen=True)
class ExperimentResult:
    seed: int
    model: str
    rmse: float
    mae: float
    mape: float
    tuning_seconds: float
    training_seconds: float
    testing_seconds: float
    best_params: dict


def run_synthetic_experiment(
    seeds: Iterable[int] = (0,),
    look_back: int = 20,
    horizon: int = 1,
    n_layers: int = 10,
    max_evals: int = 50,
    n_steps: int = 5000,
) -> list[ExperimentResult]:
    """Run synthetic EDRHN experiments without writing result artifacts."""

    results = []
    for seed in seeds:
        frame = generate_synthetic_volatility(n_steps=n_steps, seed=seed)
        x, y = make_supervised_frame(frame, look_back=look_back, horizon=horizon)
        split = chronological_split(len(x))
        train, validation, full_train, test, inverse_y = minmax_scale_splits(x, y, split)
        tune_x = np.vstack([train[0], validation[0]])
        tune_y = np.concatenate([train[1], validation[1]])

        start = perf_counter()
        tuned = tune_edrhn(
            tune_x,
            tune_y,
            n_layers=n_layers,
            validation_fraction=len(validation[1]) / len(tune_y),
            max_evals=max_evals,
            random_state=seed,
            refit=False,
        )
        tuning_seconds = perf_counter() - start

        start = perf_counter()
        model = EDRHNRegressor(
            n_layers=n_layers,
            layer_params=tuned.best_params["layer_params"],
            random_state=seed,
        ).fit(*full_train)
        training_seconds = perf_counter() - start
        results.append(
            _evaluate_model(
                seed,
                "EDRHN",
                model,
                test,
                inverse_y,
                tuning_seconds=tuning_seconds,
                training_seconds=training_seconds,
                best_params=tuned.best_params,
            )
        )
    return results


def _evaluate_model(seed, name, model, test, inverse_y, tuning_seconds, training_seconds, best_params):
    start = perf_counter()
    pred = model.predict(test[0])
    testing_seconds = perf_counter() - start
    truth = inverse_y(test[1])
    pred = inverse_y(pred)
    return ExperimentResult(
        seed=seed,
        model=name,
        rmse=root_mean_squared_error(truth, pred),
        mae=mean_absolute_error(truth, pred),
        mape=mean_absolute_percentage_error(truth, pred),
        tuning_seconds=tuning_seconds,
        training_seconds=training_seconds,
        testing_seconds=testing_seconds,
        best_params=best_params,
    )
