[![Feature - Volatility Forecasting](https://img.shields.io/badge/Feature-Volatility%20Forecasting-blue)](https://github.com/statsdl/volatility-edrhn)
[![GitHub last commit](https://img.shields.io/github/last-commit/statsdl/volatility-edrhn)](https://github.com/statsdl/volatility-edrhn/commits/main)
[![GitHub issues](https://img.shields.io/github/issues/statsdl/volatility-edrhn)](https://github.com/statsdl/volatility-edrhn/issues)
[![GitHub stars](https://img.shields.io/github/stars/statsdl/volatility-edrhn)](https://github.com/statsdl/volatility-edrhn/stargazers)
[![Python Version](https://img.shields.io/pypi/pyversions/volatility-edrhn)](https://pypi.org/project/volatility-edrhn/)
[![License](https://img.shields.io/pypi/l/volatility-edrhn)](https://github.com/statsdl/volatility-edrhn/blob/main/LICENSE)

A rich documentation is available in the GitHub repository.

# volatility-edrhn

Ensemble Deep Randomized Hopfield Network for volatility time-series forecasting.

volatility-edrhn is a Python package for volatility forecasting using ensemble deep randomized Hopfield network models with multiple output layers. 

edrhn combines randomized neural feature mapping, Hopfield-style associative memory transformations, and efficient regularized output-layer learning. This allows the model to capture nonlinear volatility dynamics without requiring expensive gradient-based training of all hidden parameters.

## Key Features

- **EDRHN Model**: Implements an ensemble deep randomized Hopfield network for volatility forecasting.
- **Multiple Output Layers**: Uses layer-wise output readouts and aggregates predictions across hidden depths.
- **Hopfield-Style Feature Interaction**: Randomized Hopfield transformations help capture nonlinear dependence in volatility features.
- **Efficient Training**: Uses randomized hidden representations with regularized output-layer learning.
- **Hyperparameter Tuning**: Supports Hyperopt/TPE search for layer-wise model selection.
- **Forecasting Metrics**: Provides RMSE, MAE, and MAPE helpers.

## Installation

Downloading locally and installing:

```bash
git clone https://github.com/statsdl/volatility-edrhn.git
cd volatility-edrhn
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install the package:

```bash
pip install -e .
```

Using pip install from GitHub:

```bash
pip install git+https://github.com/statsdl/volatility-edrhn.git
```

Using pip install from PyPI:

```bash
pip install volatility-edrhn
```

Optional tuning installation:

```bash
pip install "volatility-edrhn[tuning]"
```

Development installation:

```bash
pip install -e ".[dev]"
```

## Usage

### 1. EDRHNRegressor

```python
from volatility_edrhn import (
    EDRHNRegressor,
    generate_synthetic_volatility,
    make_supervised_frame,
)

frame = generate_synthetic_volatility(n_steps=1200, seed=0)
X, y = make_supervised_frame(frame, look_back=20, horizon=1)

model = EDRHNRegressor(n_layers=3, random_state=0)
model.fit(X[:800], y[:800])

pred = model.predict(X[800:])
print("Volatility forecasts:", pred[:5])
```

### 2. Hyperopt/TPE tuning

```python
from volatility_edrhn import generate_synthetic_volatility, make_supervised_frame
from volatility_edrhn.tuning import tune_edrhn

frame = generate_synthetic_volatility(n_steps=1200, seed=0)
X, y = make_supervised_frame(frame, look_back=20, horizon=1)

result = tune_edrhn(
    X[:900],
    y[:900],
    n_layers=3,
    max_evals=10,
    random_state=0,
)

print("Best validation loss:", result.validation_loss)
print("Best parameters:", result.best_params)
```

### 3. Synthetic benchmark

```bash
python examples/run_synthetic.py --seeds 0 --layers 3 --evals 10
```

The benchmark reports RMSE, MAE, MAPE, tuning time, training time, testing time, and selected hyperparameters.

## API Reference

### EDRHNRegressor

An ensemble deep randomized Hopfield network regressor for volatility forecasting.

**Parameters**

- `n_layers` (int): Number of stacked randomized Hopfield layers.
- `layer_params` (list): Optional layer-wise parameter dictionaries.
- `random_state` (int): Random seed for reproducibility.
- `aggregation` (str): Prediction aggregation method, either `median` or `mean`.

**Methods**

- `fit(X, y)`: Fits the ensemble model.
- `predict(X)`: Generates final volatility forecasts.
- `predict(X, return_layers=True)`: Returns layer-wise predictions.

### Data utilities

- `generate_synthetic_volatility`: Generates synthetic volatility data with volatility clustering.
- `make_supervised_frame`: Converts a volatility frame into supervised lagged arrays.
- `chronological_split`: Creates chronological train, validation, full-train, and test indexes.

### Tuning utilities

- `edrhn_search_space`: Returns the default Hyperopt/TPE search space.
- `tune_edrhn`: Tunes EDRHN hyperparameters.

### Metrics

- `root_mean_squared_error`: Computes RMSE.
- `mean_absolute_error`: Computes MAE.
- `mean_absolute_percentage_error`: Computes MAPE.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@article{bhambu2025randomized,
  title={Randomized deep Hopfield network with multiple output layers for volatility time series forecasting},
  author={Bhambu, Aryan and Natarajan, Selvaraju and Suganthan, Ponnuthurai Nagaratnam},
  journal={Neural Networks},
  pages={108207},
  year={2025},
  doi={10.1016/j.neunet.2025.108207}
}
```
