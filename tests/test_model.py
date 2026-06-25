import numpy as np

from volatility_edrhn import EDRHNRegressor


def test_edrhn_fit_predict_shapes():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(80, 6))
    y = x[:, 0] * 0.2 - x[:, 1] * 0.1

    model = EDRHNRegressor(n_layers=2, random_state=0)
    model.fit(x[:60], y[:60])

    pred = model.predict(x[60:])
    layers = model.predict(x[60:], return_layers=True)

    assert pred.shape == (20,)
    assert layers.shape == (2, 20)
