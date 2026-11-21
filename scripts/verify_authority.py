"""Verify the packaged protocol, raw journals, selection and published decisions."""

import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alignment_metaplasticity.authority import strict_json
from experiments.authority_analysis import analyse, journal, validate
from experiments.authority_data import digest, preflight
from experiments.authority_study import selection
from scripts.verify_calibration import equivalent


def verify(root=ROOT):
    directory = root / "results/authority-v1"
    checks = strict_json((directory / "checksums.json").read_text())
    expected = {
        "development.jsonl.gz",
        "test.jsonl.gz",
        "lock.json",
        "summary.json",
        "resources.json",
    }
    if set(checks) != expected:
        raise ValueError("unexpected evidence manifest")
    for name, expected_hash in checks.items():
        path = directory / name
        if (
            path.is_symlink()
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash
        ):
            raise ValueError(f"evidence digest mismatch for {name}")
    plan = strict_json((root / "studies/authority-v1/protocol.json").read_text())
    lock = strict_json((directory / "lock.json").read_text())
    development = journal(directory / "development.jsonl.gz")
    validate(plan, development, "development", plan["development_blocks"])
    chosen = selection(plan, development)
    if (
        digest(plan) != lock["protocol"]
        or any(lock[k] != v for k, v in chosen.items())
        or hashlib.sha256(
            gzip.decompress((directory / "development.jsonl.gz").read_bytes())
        ).hexdigest()
        != lock["development_sha256"]
    ):
        raise ValueError("development evidence disagrees with release lock")
    blocks = [
        (phase, i)
        for phase, n in (
            ("development", plan["development_blocks"]),
            ("test", chosen["blocks"]),
        )
        for i in range(n)
    ]
    if lock["corpus_preflight"] != preflight(blocks):
        raise ValueError("corpus stream identity mismatch")
    test = journal(directory / "test.jsonl.gz")
    result = analyse(plan, lock, test)
    if not equivalent(result, strict_json((directory / "summary.json").read_text())):
        raise ValueError("reported outcomes differ from rescored predictions")
    return {
        "status": "PASS",
        "development_attempts": len(development),
        "held_out_attempts": len(test),
        "families": list(plan["models"]),
        "blocks_per_family": lock["blocks"],
        "benchmark_gate": result["benchmark_gate"],
        "method_gate": result["method_gate"],
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
