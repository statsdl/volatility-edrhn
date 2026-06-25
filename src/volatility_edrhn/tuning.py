from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import root_mean_squared_error
from .model import EDRHNLayerParams, EDRHNRegressor


@dataclass(frozen=True)
class TuningResult:
    best_params: dict
    validation_loss: float
    trials: object
    model: EDRHNRegressor | None = None


def edrhn_search_space(prefix: str = "edrhn"):
    """Return the paper-aligned EDRHN Hyperopt search space."""

    from hyperopt import hp

    return {
        "hidden_nodes": hp.quniform(f"{prefix}_hidden_nodes", 8, 2000, 1),
        "input_scale": hp.uniform(f"{prefix}_input_scale", 0.0, 1.0),
        "regularization": hp.uniform(f"{prefix}_regularization", 0.0, 1.0),
        "ratio": hp.uniform(f"{prefix}_ratio", 0.1, 1.0),
        "hopfield_hidden_size": hp.randint(f"{prefix}_hopfield_hidden_size", 8, 100),
        "num_heads": hp.quniform(f"{prefix}_num_heads", 1, 12, 2),
        "hopfield_update_steps": hp.randint(f"{prefix}_hopfield_update_steps", 2, 6),
        "hopfield_scaling": hp.uniform(f"{prefix}_hopfield_scaling", 0.1, 0.4),
    }


def normalize_edrhn_params(params: dict) -> dict:
    def value(name):
        if name in params:
            return params[name]
        matches = [item for key, item in params.items() if key.endswith(f"_{name}")]
        if not matches:
            raise KeyError(name)
        return matches[0]

    return {
        "hidden_nodes": int(value("hidden_nodes")),
        "input_scale": float(value("input_scale")),
        "regularization": max(float(value("regularization")), 1.0e-8),
        "ratio": float(value("ratio")),
        "hopfield_hidden_size": int(value("hopfield_hidden_size")),
        "num_heads": max(1, int(value("num_heads"))),
        "hopfield_update_steps": int(value("hopfield_update_steps")),
        "hopfield_scaling": float(value("hopfield_scaling")),
    }




def tune_edrhn(
    x,
    y,
    n_layers: int = 10,
    validation_fraction: float = 0.1 / 0.8,
    max_evals: int = 50,
    random_state: int = 0,
    refit: bool = True,
) -> TuningResult:
    """Layerwise Hyperopt/TPE tuning for EDRHN."""

    from hyperopt import STATUS_OK, Trials, fmin, space_eval, tpe

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")
    split_at = int(round(len(x) * (1.0 - validation_fraction)))
    train_x, val_x = x[:split_at], x[split_at:]
    train_y, val_y = y[:split_at], y[split_at:]
    if len(val_y) == 0:
        raise ValueError("validation split is empty.")

    layer_params = []
    losses = []
    all_trials = []
    for layer_idx in range(n_layers):
        trials = Trials()

        def objective(sample):
            params = normalize_edrhn_params(sample)
            model = EDRHNRegressor(
                n_layers=layer_idx + 1,
                layer_params=layer_params + [params],
                random_state=random_state,
            )
            model.fit(train_x, train_y)
            prediction = model.predict(val_x)
            return {"loss": root_mean_squared_error(val_y, prediction), "status": STATUS_OK}

        best = fmin(
            fn=objective,
            space=edrhn_search_space(prefix=f"layer_{layer_idx + 1}"),
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            rstate=np.random.default_rng(random_state + layer_idx),
            show_progressbar=False,
        )
        best_params = normalize_edrhn_params(best)
        layer_params.append(best_params)
        losses.append(float(trials.best_trial["result"]["loss"]))
        all_trials.append(trials)

    best_model = None
    if refit:
        best_model = EDRHNRegressor(
            n_layers=n_layers,
            layer_params=layer_params,
            random_state=random_state,
        ).fit(x, y)

    return TuningResult(
        best_params={"layer_params": layer_params, "random_state": random_state},
        validation_loss=float(losses[-1]),
        trials=all_trials,
        model=best_model,
    )
