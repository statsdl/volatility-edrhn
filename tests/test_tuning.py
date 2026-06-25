from volatility_edrhn.data import generate_synthetic_volatility, make_supervised_frame
from volatility_edrhn.tuning import edrhn_search_space, tune_edrhn


def test_search_space_keys():
    assert sorted(edrhn_search_space()) == [
        "hidden_nodes",
        "hopfield_hidden_size",
        "hopfield_scaling",
        "hopfield_update_steps",
        "input_scale",
        "num_heads",
        "ratio",
        "regularization",
    ]


def test_hyperopt_tuning_smoke():
    frame = generate_synthetic_volatility(n_steps=180, seed=0)
    x, y = make_supervised_frame(frame, look_back=10, horizon=1)
    result = tune_edrhn(x[:100], y[:100], n_layers=1, max_evals=2, random_state=0)

    assert "layer_params" in result.best_params
    assert len(result.best_params["layer_params"]) == 1
