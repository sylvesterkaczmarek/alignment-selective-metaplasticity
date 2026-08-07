from ._common import run_named

if __name__ == "__main__":
    run_named("configs/baseline.yaml", "baseline_forgetting", methods=("fine_tune",), include_bypass=False)
