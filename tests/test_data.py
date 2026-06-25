from volatility_edrhn import chronological_split, generate_synthetic_volatility, make_supervised_frame


def test_synthetic_frame_and_split():
    frame = generate_synthetic_volatility(n_steps=200, seed=0)
    x, y = make_supervised_frame(frame, look_back=20, horizon=1)
    split = chronological_split(len(x))

    assert x.shape[0] == y.shape[0]
    assert x.shape[1] == 80
    assert len(split.train) + len(split.validation) + len(split.test) == len(x)
    assert len(split.full_train) == len(split.train) + len(split.validation)
