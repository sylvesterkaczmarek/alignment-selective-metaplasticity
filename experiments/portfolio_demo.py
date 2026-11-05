"""Train the four reference methods on one seed and display the retention trade-off."""
from __future__ import annotations
import argparse
from pathlib import Path
from alignment_metaplasticity.experiment import run_suite
from alignment_metaplasticity.io_utils import write_json
from alignment_metaplasticity.summary import summarize_suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    suite = run_suite(None, seeds=[7])
    write_json(args.out/'runs.json', suite)
    write_json(args.out/'summary.json', summarize_suite(suite))
    print('One-seed synthetic demonstration. No population-level safety claim.')
    print(f'{"method":18} {"alignment":>10} {"acquisition":>12} {"bypass":>8}')
    for row in suite['runs']:
        print(f'{row["method"]:18} {row["final_alignment"]["selective_corrigibility"]:10.3f} '
              f'{row["mean_acquisition_accuracy"]:12.3f} {row["bypass"]["bypass_success_rate"]:8.3f}')
    print('Bypass training includes ordinary-alignment rehearsal. Endpoint retention cannot establish continuous preservation.')


if __name__ == '__main__':
    main()
