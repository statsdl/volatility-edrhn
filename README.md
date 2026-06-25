# volatility-edrhn

[![Tests](https://github.com/statsdl/volatility-edrhn/actions/workflows/test.yml/badge.svg)](https://github.com/statsdl/volatility-edrhn/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/volatility-edrhn)](https://pypi.org/project/volatility-edrhn/)
[![Python Version](https://img.shields.io/pypi/pyversions/volatility-edrhn)](https://pypi.org/project/volatility-edrhn/)
[![License](https://img.shields.io/github/license/statsdl/volatility-edrhn)](LICENSE)

**Ensemble Deep Randomized Hopfield Network for volatility time-series forecasting.**

volatility-edrhn provides an implementation of EDRHN-style volatility forecasting models with randomized Hopfield transformations, multiple output layers, Hyperopt tuning utilities, and synthetic benchmark helpers.

## Use Cases

- Financial volatility forecasting
- Randomized neural network experiments
- Synthetic volatility benchmarks
- Time-series regression research
- Volatility risk-model prototyping

## Key Features

- EDRHNRegressor with stacked randomized Hopfield-style layers
- Multiple output-layer aggregation using median or mean readouts
- Synthetic volatility data generation
- Chronological train-validation-test splitting
- Hyperopt/TPE tuning support
- RMSE, MAE, and MAPE forecasting metrics
- No bundled private datasets or generated result artifacts

## Installation

```bash
pip install volatility-edrhn
```

From source:

```bash
git clone https://github.com/statsdl/volatility-edrhn.git
cd volatility-edrhn
pip install -e .
```

For tuning:

```bash
pip install "volatility-edrhn[tuning]"
```

For development:

```bash
pip install -e ".[dev]"
pytest
```

## Quick Start

```python
from volatility_edrhn import EDRHNRegressor, generate_synthetic_volatility, make_supervised_frame

frame = generate_synthetic_volatility(n_steps=1200, seed=0)
X, y = make_supervised_frame(frame, look_back=20, horizon=1)

model = EDRHNRegressor(n_layers=3, random_state=0)
model.fit(X[:800], y[:800])
pred = model.predict(X[800:])

print(pred[:5])
```

## Synthetic Experiment

```bash
python examples/run_synthetic.py --seeds 0 --layers 3 --evals 10
```

The example prints RMSE, MAE, MAPE, tuning time, training time, testing time, and selected hyperparameters.

## Hyperopt Tuning

```python
from volatility_edrhn import generate_synthetic_volatility, make_supervised_frame
from volatility_edrhn.tuning import tune_edrhn

frame = generate_synthetic_volatility(n_steps=1200, seed=0)
X, y = make_supervised_frame(frame, look_back=20, horizon=1)

result = tune_edrhn(X[:900], y[:900], n_layers=3, max_evals=10, random_state=0)
print(result.validation_loss)
```

## API

### Main model

- `EDRHNRegressor`: Ensemble deep randomized Hopfield network regressor.
- `EDRHNLayerParams`: Layer-wise parameter container.

### Data utilities

- `generate_synthetic_volatility`
- `make_supervised_frame`
- `chronological_split`

### Metrics

- `root_mean_squared_error`
- `mean_absolute_error`
- `mean_absolute_percentage_error`

## Repository Layout

```text
src/volatility_edrhn/   Supported package API
examples/               Usage examples
tests/                  Unit tests
docs/                   Documentation stubs
.github/                CI and PyPI publishing workflows
```

## Citation

If you use volatility-edrhn in your research, please cite:

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

## Authors

- Aryan Bhambu
- Selvaraju Natarajan
- Ponnuthurai Nagaratnam Suganthan

## License

MIT License. See [LICENSE](LICENSE) for details.
