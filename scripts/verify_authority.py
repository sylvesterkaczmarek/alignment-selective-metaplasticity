"""Verify the packaged protocol, raw journals, selection and published decisions."""

import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alignment_metaplasticity.authority import strict_json
from experiments.authority_analysis import (
    OPEN_WEIGHT_FILES,
    analyse,
    check_evidence_digests,
    journal,
    validate,
)
from experiments.authority_data import digest, preflight
from experiments.authority_study import selection
from scripts.verify_calibration import equivalent


def recount_synthetic(plan, rows, release_floor, failure_floor):
    planned = {
        (condition, digest(trial), method, protocol)
        for condition in plan["conditions"]
        for trial in plan["trials"]
        for method, protocol in plan["methods"]
    }
    remaining = planned.copy()
    qualified = completed = 0
    events = []
    for row in rows:
        key = (row["condition"], digest(row["trial"]), row["method"], row["protocol"])
        if key not in remaining:
            raise ValueError("duplicate or unplanned synthetic trial")
        remaining.remove(key)
        if row["status"] == "failed" and row.get("error"):
            continue
        if row["status"] != "completed":
            raise ValueError("invalid synthetic terminal status")
        tasks = len(row["trial"]["order"])
        epochs = plan["config"]["training"]["capability_epochs"]
        steps = 1 + tasks * (epochs + int(row["method"] == "rehearsal"))
        trajectory = row["trajectory"]
        if len(trajectory) != steps or any(
            not isinstance(point, list)
            or len(point) != 2
            or any(type(x) not in (int, float) or not 0 <= x <= 1 for x in point)
            for point in trajectory
        ):
            raise ValueError("invalid synthetic checkpoint trajectory")
        completed += 1
        if min(trajectory[0]) >= release_floor:
            qualified += 1
            if min(trajectory[-1]) >= release_floor and any(
                min(point) < failure_floor for point in trajectory[1:-1]
            ):
                events.append(
                    {
                        "condition": row["condition"],
                        "model_seed": row["trial"]["model_seed"],
                        "method": row["method"],
                        "protocol": row["protocol"],
                    }
                )
    if remaining:
        raise ValueError("missing planned synthetic trial")
    return {
        "attempts": len(planned),
        "completed": completed,
        "baseline_qualified": qualified,
        "hidden_retention_failures": len(events),
        "events": events,
    }


def verify(root=ROOT):
    directory = root / "results/authority-v1"
    check_evidence_digests(directory, OPEN_WEIGHT_FILES)
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
    controls = root / "results/authority-synthetic-v1"
    check_evidence_digests(controls, ("plan.json", "runs.jsonl.gz"))
    synthetic = recount_synthetic(
        strict_json((controls / "plan.json").read_text()),
        journal(controls / "runs.jsonl.gz"),
        plan["release_floor"],
        plan["failure_floor"],
    )
    return {
        "status": "PASS",
        "development_attempts": len(development),
        "held_out_attempts": len(test),
        "families": list(plan["models"]),
        "blocks_per_family": lock["blocks"],
        "baseline_qualified_blocks": {
            family: stats["methods"]["unprotected"]["baseline_pass"]
            for family, stats in result["families"].items()
        },
        "synthetic_control": synthetic,
        "benchmark_gate": result["benchmark_gate"],
        "method_gate": result["method_gate"],
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
