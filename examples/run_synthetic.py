import argparse

from volatility_edrhn.experiment import run_synthetic_experiment


def parse_seeds(value):
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main():
    parser = argparse.ArgumentParser(description="Run synthetic EDRHN volatility experiments.")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--look-back", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--layers", type=int, default=10)
    parser.add_argument("--evals", type=int, default=50)
    parser.add_argument("--steps", type=int, default=5000)
    args = parser.parse_args()

    results = run_synthetic_experiment(
        seeds=parse_seeds(args.seeds),
        look_back=args.look_back,
        horizon=args.horizon,
        n_layers=args.layers,
        max_evals=args.evals,
        n_steps=args.steps,
    )
    for result in results:
        print(
            f"seed={result.seed} {result.model} "
            f"RMSE={result.rmse:.6f} MAE={result.mae:.6f} MAPE={result.mape:.6f} "
            f"tune={result.tuning_seconds:.3f}s train={result.training_seconds:.3f}s "
            f"test={result.testing_seconds:.3f}s"
        )


if __name__ == "__main__":
    main()
