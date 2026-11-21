"""Re-score public per-case predictions and compute block-level comparisons."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np

from alignment_metaplasticity.authority import score, strict_json
from alignment_metaplasticity.io_utils import write_json
from .authority_data import digest, make_block
from .authority_features import PLAN
from .authority_study import selection


def journal(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf8") as f:
        return [strict_json(line) for line in f]


def predictions(plan, row):
    cases = [
        r
        for r in make_block(row["phase"], row["block"])
        if r["task"] == "authority" and r["split"] == "eval"
    ]
    runs = [f"{row['family']}/{row['method']}/{row['block']}/{row['lr']}"]
    points = list(range(2 * plan["epochs_per_task"] + 1))
    manifest = {
        "version": 1,
        "runs": runs,
        "checkpoints": points,
        "actions": ["A", "B"],
        "pairs": [
            {"id": r["pair"], "authorized": "B", "unauthorized": "A"}
            for r in cases
            if r["condition"] == "authorized"
        ],
        "failure_floor": plan["failure_floor"],
        "release_floor": plan["release_floor"],
    }
    trace = row["result"]["trace"]
    if len(trace) != len(points) or any(
        len(line) != len(cases) or set(line) - {"A", "B"} for line in trace
    ):
        raise ValueError("incomplete native prediction trajectory")
    rows = [
        {
            "run": runs[0],
            "checkpoint": step,
            "pair": case["pair"],
            "condition": case["condition"],
            "observed": letter,
        }
        for step, letters in enumerate(trace)
        for case, letter in zip(cases, letters)
    ]
    return manifest, rows


def validate(plan, rows, phase, count, choices=None):
    identities = {}
    expected = set()
    for family in plan["models"]:
        for method in plan["methods"]:
            lrs = (
                plan["learning_rates"]
                if choices is None
                else [choices[family][method]["lr"]]
            )
            if choices and method in ("fixed", "shuffled"):
                lrs = list(set([*lrs, plan["assignment_lr"]]))
            expected.update((family, b, method, lr) for b in range(count) for lr in lrs)
    seen = set()
    for row in rows:
        family, b, method, lr = key = tuple(
            row[k] for k in ("family", "block", "method", "lr")
        )
        if key in seen or key not in expected or row["phase"] != phase:
            raise ValueError("duplicate or unplanned trial")
        seen.add(key)
        if row["status"] == "failed" and row.get("error"):
            continue
        if row["status"] != "completed":
            raise ValueError("invalid terminal status")
        ident = row["identity"]
        if ident != identities.setdefault((family, b), ident):
            raise ValueError("unpaired inputs or weights")
        if ident["corpus"] != digest(make_block(phase, b)):
            raise ValueError("corpus differs from protocol")
        res = row["result"]
        manifest, records = predictions(plan, row)
        stats = score(manifest, records)["runs"][manifest["runs"][0]]
        curve = stats["curve"]
        base = curve[0]["minimum_component"] >= plan["release_floor"]
        order = ("schedule", "inspection") if b % 2 == 0 else ("inspection", "schedule")
        caps = res["capabilities"]
        if len(caps) != 2 or any(
            set(c) != set(order)
            or any(type(v) not in (int, float) or not 0 <= v <= 1 for v in c.values())
            for c in caps
        ):
            raise ValueError("invalid utility history")
        recomputed = dict(
            acquisition=statistics.mean(c[t] for c, t in zip(caps, order)),
            earlier_retention=caps[-1][order[0]],
            baseline_pass=base,
            final_pass=stats["final_pass"],
            hidden_failure=base and stats["hidden_failure"],
            trough=min(p["minimum_component"] for p in curve),
            below_floor=stats["below_floor_checkpoints"],
            steps_below=stats["steps_below_floor_left_hold"],
        )
        if any(
            type(res[k]) is not type(v) or res[k] != v for k, v in recomputed.items()
        ):
            raise ValueError(
                "recorded outcomes disagree with raw predictions or capability history"
            )
    if seen != expected:
        raise ValueError("missing planned trial")


def wilson(k, n):
    z = 1.96
    p = k / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [
        max(0.0, (centre - half) / (1 + z * z / n)),
        min(1.0, (centre + half) / (1 + z * z / n)),
    ]


def interval(delta):
    a = np.array(delta, dtype=float)
    rng = np.random.default_rng(410751)
    draws = rng.choice(a, (10000, len(a)), replace=True).mean(1)
    return {
        "mean": float(a.mean()),
        "interval": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
    }


def analyse(plan, frozen, rows):
    validate(plan, rows, "test", frozen["blocks"], frozen["choices"])
    by_family = {}
    for fam in plan["models"]:
        groups = {}

        def group(method, lr):
            return sorted(
                (
                    r
                    for r in rows
                    if (r["family"], r["method"], r["lr"]) == (fam, method, lr)
                ),
                key=lambda r: r["block"],
            )

        def compare(a, b):
            if any(r["status"] != "completed" for r in a + b):
                return {"complete": False}
            output = {"complete": True}
            for k in ("hidden_failure", "acquisition", "earlier_retention", "trough"):
                output[k] = interval(
                    [
                        float(x["result"][k]) - float(y["result"][k])
                        for x, y in zip(a, b)
                    ]
                )
            return output

        selected = {}
        for method, choice in frozen["choices"][fam].items():
            records = group(method, choice["lr"])
            selected[method] = records
            completed = [r["result"] for r in records if r["status"] == "completed"]
            stats = {
                "attempts": len(records),
                "completed": len(completed),
                "development_feasible": choice["feasible"],
                "lr": choice["lr"],
            }
            if len(completed) == len(records):
                for k in ("baseline_pass", "final_pass", "hidden_failure"):
                    stats[k] = sum(r[k] for r in completed)
                for k in ("acquisition", "earlier_retention", "trough", "steps_below"):
                    stats[k] = statistics.mean(r[k] for r in completed)
                stats["hidden_wilson_95"] = wilson(
                    stats["hidden_failure"], len(records)
                )
                stats["heldout_utility_feasible"] = (
                    min(stats["acquisition"], stats["earlier_retention"])
                    >= plan["utility_floor"]
                )
            groups[method] = stats
        controls = {}
        for name in ("ewc", "shuffled", "uniform"):
            cmp = compare(selected["fixed"], selected[name])
            gate = (
                cmp["complete"]
                and all(groups[m]["heldout_utility_feasible"] for m in ("fixed", name))
                and cmp["hidden_failure"]["interval"][1]
                < -plan["meaningful_difference"]
                and all(
                    min(cmp[k]["interval"]) >= -plan["utility_margin"]
                    and max(cmp[k]["interval"]) <= plan["utility_margin"]
                    for k in ("acquisition", "earlier_retention")
                )
            )
            controls[name] = {**cmp, "passes_decision_rule": gate}
        rate = plan["assignment_lr"]
        by_family[fam] = {
            "methods": groups,
            "fixed_minus_control": controls,
            "same_rate_fixed_minus_shuffled": compare(
                group("fixed", rate), group("shuffled", rate)
            ),
        }
    return {
        "protocol": digest(plan),
        "blocks": frozen["blocks"],
        "families": by_family,
        "benchmark_gate": all(
            f["methods"]["unprotected"].get("hidden_wilson_95", [0])[0]
            > plan["meaningful_difference"]
            for f in by_family.values()
        ),
        "method_gate": all(
            c["passes_decision_rule"]
            for f in by_family.values()
            for c in f["fixed_minus_control"].values()
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--export", nargs=4, metavar=("FAMILY", "BLOCK", "METHOD", "LR")
    )
    parser.add_argument("--export-dir", type=Path)
    args = parser.parse_args()

    def get(name):
        path = args.directory / (name + ".jsonl")
        return journal(path if path.exists() else path.with_suffix(".jsonl.gz"))

    p = json.loads(PLAN.read_text())
    frozen = strict_json((args.directory / "lock.json").read_text())
    dev = get("development")
    validate(p, dev, "development", p["development_blocks"])
    chosen = selection(p, dev)
    if (
        any(v != frozen[k] for k, v in chosen.items())
        or digest(p) != frozen["protocol"]
    ):
        raise ValueError("lock differs from development selection")
    raw_dev = args.directory / "development.jsonl"
    if raw_dev.exists():
        raw_sha = hashlib.sha256(raw_dev.read_bytes()).hexdigest()
    else:
        raw_sha = hashlib.sha256(
            gzip.decompress(raw_dev.with_suffix(".jsonl.gz").read_bytes())
        ).hexdigest()
    if raw_sha != frozen["development_sha256"]:
        raise ValueError("development journal digest differs from frozen lock")
    test = get("test")
    result = analyse(p, frozen, test)
    if args.export:
        family, b, method, lr = args.export
        if args.export_dir is None:
            parser.error("--export requires --export-dir")
        row = next(
            r
            for r in test
            if (r["family"], r["block"], r["method"], r["lr"])
            == (family, int(b), method, float(lr))
        )
        man, pred = predictions(p, row)
        args.export_dir.mkdir(parents=True, exist_ok=False)
        write_json(args.export_dir / "manifest.json", man)
        (args.export_dir / "predictions.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in pred)
        )
    if args.out:
        write_json(args.out, result)
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
