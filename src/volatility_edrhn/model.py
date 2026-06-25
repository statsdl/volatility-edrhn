from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EDRHNLayerParams:
    hidden_nodes: int = 100
    input_scale: float = 0.5
    regularization: float = 1.0e-3
    ratio: float = 0.5
    hopfield_hidden_size: int = 32
    num_heads: int = 1
    hopfield_update_steps: int = 2
    hopfield_scaling: float = 0.2


class EDRHNRegressor:
    """Ensemble deep randomized Hopfield network with ridge readouts."""

    def __init__(
        self,
        n_layers: int = 10,
        layer_params: list[dict | EDRHNLayerParams] | None = None,
        random_state: int | None = None,
        aggregation: str = "median",
    ):
        if n_layers <= 0:
            raise ValueError("n_layers must be positive.")
        if aggregation not in {"median", "mean"}:
            raise ValueError("aggregation must be 'median' or 'mean'.")
        self.n_layers = n_layers
        self.layer_params = layer_params
        self.random_state = random_state
        self.aggregation = aggregation

    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if x.ndim != 2:
            raise ValueError("x must be a 2D array.")
        if len(x) != len(y):
            raise ValueError("x and y must have matching rows.")

        rng = np.random.default_rng(self.random_state)
        current = x
        self.layers_ = []
        params = self._resolved_layer_params()
        for layer_idx, layer_params in enumerate(params):
            layer_rng = np.random.default_rng(rng.integers(0, np.iinfo(np.int32).max))
            transformed, state = _edrhn_features(current, layer_params, layer_rng, fit=True)
            weights = _ridge_weights(transformed, y, layer_params.regularization)
            self.layers_.append({**state, "params": layer_params, "weights": weights})
            current = transformed
        return self

    def predict(self, x, return_layers: bool = False):
        if not hasattr(self, "layers_"):
            raise RuntimeError("model is not fitted.")
        current = np.asarray(x, dtype=float)
        predictions = []
        for state in self.layers_:
            current = _apply_edrhn_features(current, state)
            predictions.append(_predict_ridge(current, state["weights"]))
        stacked = np.vstack(predictions)
        if return_layers:
            return stacked
        if self.aggregation == "mean":
            return stacked.mean(axis=0)
        return np.median(stacked, axis=0)

    def _resolved_layer_params(self) -> list[EDRHNLayerParams]:
        if self.layer_params is None:
            return [EDRHNLayerParams() for _ in range(self.n_layers)]
        resolved = []
        for item in self.layer_params:
            if isinstance(item, EDRHNLayerParams):
                resolved.append(item)
            else:
                resolved.append(EDRHNLayerParams(**item))
        if len(resolved) != self.n_layers:
            raise ValueError("layer_params length must match n_layers.")
        return resolved


def _edrhn_features(x, params: EDRHNLayerParams, rng, fit: bool):
    projection = rng.normal(
        0.0,
        max(params.input_scale, 1.0e-8),
        size=(x.shape[1], params.hidden_nodes),
    )
    bias = rng.uniform(-params.input_scale, params.input_scale, size=params.hidden_nodes)
    hidden = np.tanh(x @ projection + bias)

    memory_projection = rng.normal(
        0.0,
        params.hopfield_scaling,
        size=(params.hidden_nodes, params.hopfield_hidden_size),
    )
    memory = hidden @ memory_projection
    for _ in range(params.hopfield_update_steps):
        memory = np.tanh(memory * params.hopfield_scaling)

    keep = max(1, int(round(params.hidden_nodes * params.ratio)))
    selected = np.argsort(np.var(hidden, axis=0))[-keep:]
    features = np.hstack([x, hidden[:, selected], memory])
    state = {
        "projection": projection,
        "bias": bias,
        "memory_projection": memory_projection,
        "selected": selected,
    }
    return features, state


def _apply_edrhn_features(x, state):
    params = state["params"]
    hidden = np.tanh(x @ state["projection"] + state["bias"])
    memory = hidden @ state["memory_projection"]
    for _ in range(params.hopfield_update_steps):
        memory = np.tanh(memory * params.hopfield_scaling)
    return np.hstack([x, hidden[:, state["selected"]], memory])


def _ridge_weights(x, y, regularization: float):
    x_aug = np.hstack([np.ones((x.shape[0], 1)), x])
    identity = np.eye(x_aug.shape[1])
    identity[0, 0] = 0.0
    return np.linalg.solve(x_aug.T @ x_aug + regularization * identity, x_aug.T @ y)


def _predict_ridge(x, weights):
    x_aug = np.hstack([np.ones((x.shape[0], 1)), x])
    return x_aug @ weights
