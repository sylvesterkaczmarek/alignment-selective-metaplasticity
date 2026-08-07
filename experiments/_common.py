from __future__ import annotations

import argparse
from pathlib import Path

from alignment_metaplasticity.experiment import METHODS, run_suite
from alignment_metaplasticity.io_utils import write_json
from alignment_metaplasticity.plotting import plot_summary
from alignment_metaplasticity.summary import summarize_suite


def parse_args(default_config: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 29])
    parser.add_argument("--out", default="results")
    return parser.parse_args()


def run_named(default_config: str, stem: str, methods=METHODS, include_bypass: bool = True) -> None:
    args = parse_args(default_config)
    suite = run_suite(args.config, seeds=args.seeds, methods=tuple(methods), include_bypass=include_bypass)
    summary = summarize_suite(suite)
    out = Path(args.out)
    write_json(out / f"{stem}.json", suite)
    write_json(out / f"{stem}_summary.json", summary)
    plot_summary(summary, out / "figures")
    print(f"wrote {out / f'{stem}.json'}")
    print(f"wrote {out / f'{stem}_summary.json'}")
