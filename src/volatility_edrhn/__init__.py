"""EDRHN-style volatility forecasting utilities."""

from .data import SplitData, chronological_split, generate_synthetic_volatility, make_supervised_frame
from .metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
from .model import EDRHNLayerParams, EDRHNRegressor

__all__ = [
    "EDRHNLayerParams",
    "EDRHNRegressor",
    "SplitData",
    "chronological_split",
    "generate_synthetic_volatility",
    "make_supervised_frame",
    "mean_absolute_error",
    "mean_absolute_percentage_error",
    "root_mean_squared_error",
]

__version__ = "0.1.0"
