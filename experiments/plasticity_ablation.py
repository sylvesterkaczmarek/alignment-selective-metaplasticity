from __future__ import annotations

import argparse
from pathlib import Path

from alignment_metaplasticity.config import load_config
from alignment_metaplasticity.experiment import run_method
from alignment_metaplasticity.io_utils import write_json
from alignment_metaplasticity.plotting import plot_ablation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/metaplastic.yaml")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--strengths", nargs="+", type=float, default=[0.0, 20.0, 80.0, 200.0, 400.0])
    parser.add_argument("--out", default="results/plasticity_ablation.json")
    args = parser.parse_args()

    base = load_config(args.config)
    runs = []
    for strength in args.strengths:
        cfg = load_config(args.config, {"metaplastic": {"strength": strength}})
        run = run_method(cfg, args.seed, "metaplastic", include_bypass=False)
        run["metaplastic_strength"] = strength
        runs.append(run)
    payload = {"base_config": base, "runs": runs}
    write_json(Path(args.out), payload)
    plot_ablation(payload, Path(args.out).parent / "figures" / "plasticity_ablation.png")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
