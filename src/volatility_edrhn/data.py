from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_FEATURES = ("return", "garch_variance", "abs_return", "squared_return")
DEFAULT_TARGET = "realized_volatility"


@dataclass(frozen=True)
class SplitData:
    train: np.ndarray
    validation: np.ndarray
    full_train: np.ndarray
    test: np.ndarray


def generate_synthetic_volatility(
    n_steps: int = 5000,
    seed: int = 0,
    freq: str = "5min",
) -> pd.DataFrame:
    """Generate deterministic 5-minute-like volatility data."""

    if n_steps < 100:
        raise ValueError("n_steps must be at least 100.")

    rng = np.random.default_rng(seed)
    omega = 1.0e-6
    alpha = 0.08
    beta = 0.90
    variance = np.empty(n_steps, dtype=float)
    returns = np.empty(n_steps, dtype=float)
    variance[0] = omega / (1.0 - alpha - beta)
    returns[0] = np.sqrt(variance[0]) * rng.standard_normal()

    for idx in range(1, n_steps):
        variance[idx] = omega + alpha * returns[idx - 1] ** 2 + beta * variance[idx - 1]
        intraday = 1.0 + 0.25 * np.sin(2.0 * np.pi * (idx % 78) / 78)
        returns[idx] = np.sqrt(variance[idx] * intraday) * rng.standard_t(df=8)

    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01 09:30:00", periods=n_steps, freq=freq),
            "return": returns,
            "garch_variance": variance,
        }
    )
    frame["abs_return"] = frame["return"].abs()
    frame["squared_return"] = frame["return"] ** 2
    frame["realized_volatility"] = (
        frame["squared_return"].rolling(window=48, min_periods=1).mean().pow(0.5)
    )
    return frame


def make_supervised_frame(
    frame: pd.DataFrame,
    look_back: int = 20,
    horizon: int = 1,
    features: Iterable[str] = DEFAULT_FEATURES,
    target: str = DEFAULT_TARGET,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a volatility frame into lagged supervised arrays."""

    if look_back <= 0 or horizon <= 0:
        raise ValueError("look_back and horizon must be positive.")

    feature_columns = tuple(features)
    values = frame.loc[:, feature_columns].to_numpy(dtype=float)
    target_values = frame.loc[:, target].to_numpy(dtype=float)
    n_samples = len(frame) - look_back - horizon + 1
    if n_samples <= 0:
        raise ValueError("frame is too short for the requested look_back and horizon.")

    x = np.empty((n_samples, look_back * len(feature_columns)), dtype=float)
    y = np.empty(n_samples, dtype=float)
    for idx in range(n_samples):
        x[idx] = values[idx : idx + look_back].reshape(-1)
        y[idx] = target_values[idx + look_back + horizon - 1]
    return x, y


def chronological_split(
    n_samples: int,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.10,
) -> SplitData:
    """Return chronological 70:10:20 indexes by default."""

    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1.")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must be less than 1.")

    train_end = int(round(n_samples * train_fraction))
    validation_end = train_end + int(round(n_samples * validation_fraction))
    return SplitData(
        train=np.arange(0, train_end),
        validation=np.arange(train_end, validation_end),
        full_train=np.arange(0, validation_end),
        test=np.arange(validation_end, n_samples),
    )


def minmax_scale_splits(
    x: np.ndarray,
    y: np.ndarray,
    split: SplitData,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], callable]:
    """Scale split arrays using train-only min-max statistics."""

    x_min = x[split.train].min(axis=0)
    x_range = np.maximum(x[split.train].max(axis=0) - x_min, 1.0e-12)
    y_min = float(y[split.train].min())
    y_range = max(float(y[split.train].max() - y_min), 1.0e-12)

    def scale_x(values):
        return (values - x_min) / x_range

    def scale_y(values):
        return (values - y_min) / y_range

    def inverse_y(values):
        return np.asarray(values, dtype=float) * y_range + y_min

    return (
        (scale_x(x[split.train]), scale_y(y[split.train])),
        (scale_x(x[split.validation]), scale_y(y[split.validation])),
        (scale_x(x[split.full_train]), scale_y(y[split.full_train])),
        (scale_x(x[split.test]), scale_y(y[split.test])),
        inverse_y,
    )
