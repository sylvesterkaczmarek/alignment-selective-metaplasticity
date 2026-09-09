"""A development sweep followed by a locked, independent test evaluation."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics

from alignment_metaplasticity.config import load_config
from alignment_metaplasticity.experiment import runtime_metadata
from alignment_metaplasticity.io_utils import write_json
from alignment_metaplasticity.study import Setting, StudyRun, Trial, STUDY_METHODS


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def metadata():
    return {**runtime_metadata("cpu"), "study_driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def candidates():
    settings = []
    for protocol in ("fixed", "periodic"):
        for method in STUDY_METHODS:
            if method == "rehearsal" and protocol == "fixed":
                continue
            if method in ("fine_tune", "rehearsal"):
                settings.extend(Setting(method, protocol, lr) for lr in (.01, .02, .03, .05))
            else:
                name, values = (("ewc_lambda", (1., 18.)) if method == "ewc" else
                                ("freeze_quantile", (.5, .85)) if method == "static_freeze" else
                                ("strength", (20., 200.)))
                settings.extend(Setting(method, protocol, lr, **{name: v}) for lr in (.015, .03) for v in values)
    return [asdict(s) for s in settings]


def aggregate(results):
    metrics = {"alignment": [r["alignment"]["selective_corrigibility"] for r in results],
               "authorized_acceptance": [r["alignment"]["authorized_acceptance"] for r in results],
               "unauthorized_resistance": [r["alignment"]["unauthorized_resistance"] for r in results]}
    metrics.update({k: [r[k] for r in results] for k in ("acquisition", "retention", "seconds")})
    return {k: {"mean": statistics.mean(v), "sd": statistics.stdev(v) if len(v) > 1 else None, "n": len(v)}
            for k, v in metrics.items()}


def select(records, settings, repetitions, utility_floor):
    eligible, invalid = {}, []
    for setting in settings:
        matches = [r for r in records if r["setting"] == setting]
        if len(matches) != repetitions or any(r["status"] != "completed" for r in matches):
            invalid.append(setting)
            continue
        stats = aggregate([r["result"] for r in matches])
        group = setting["protocol"] + "/" + setting["method"]
        eligible.setdefault(group, []).append({"setting": setting, "development": stats})
    selected = []
    for group, options in sorted(eligible.items()):
        feasible = [o for o in options if o["development"]["acquisition"]["mean"] >= utility_floor]
        # An infeasible fallback is retained for diagnosis and explicitly flagged.
        chosen = max(feasible or options, key=lambda o: (o["development"]["alignment" if feasible else "acquisition"]["mean"],
                                                        o["development"]["acquisition"]["mean"]))
        selected.append({"group": group, "utility_feasible_on_development": bool(feasible), **chosen})
    return {"selected": selected, "invalid_candidates": invalid}


def execute(path, cfg, settings, trials):
    records = []
    with path.open("x", encoding="utf-8") as f:
        for setting in settings:
            for trial in trials:
                item = {"setting": setting, "trial": trial, "id": digest([setting, trial])}
                f.write(json.dumps({**item, "status": "started"}) + "\n")
                f.flush()
                try:
                    result = StudyRun(cfg, Trial(**trial), Setting(**setting)).run()
                    # Reject NaN/inf before a run becomes eligible for selection.
                    json.dumps(result, allow_nan=False)
                    item.update(status="completed", result=result)
                except Exception as exc:
                    item.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                f.write(json.dumps(item, allow_nan=False) + "\n")
                f.flush()
                records.append(item)
                print(f'{len(records)}/{len(settings)*len(trials)} {setting["protocol"]}/{setting["method"]} {item["status"]}', flush=True)
    return records


def plot(rows, path, *, frontier=False):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for row in rows:
        stats = row["metrics"]
        ax.errorbar(stats["acquisition"]["mean"], stats["alignment"]["mean"],
                    xerr=stats["acquisition"]["sd"] or 0, yerr=stats["alignment"]["sd"] or 0,
                    fmt="o", label=row["group"] if not frontier else None)
    if frontier:
        for group in sorted({r["group"] for r in rows}):
            points = [(r["metrics"]["acquisition"]["mean"], r["metrics"]["alignment"]["mean"])
                      for r in rows if r["group"] == group]
            nondominated = sorted((x, y) for x, y in points
                                  if not any(a >= x and b >= y and (a > x or b > y) for a, b in points))
            ax.plot([p[0] for p in nondominated], [p[1] for p in nondominated], "-o", label=group)
    ax.set(xlabel="New-task acquisition", ylabel="Alignment score", title=("Development" if frontier else "Held-out") + " means and between-trial SD")
    ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("development", "test"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", default="configs/metaplastic.yaml")
    args = parser.parse_args()
    out = args.out
    if args.phase == "development":
        out.mkdir(parents=True, exist_ok=False)
        cfg = load_config(args.config)
        tasks = cfg["model"]["num_capability_tasks"]
        orders = [tuple(range(1, tasks + 1)), tuple(range(tasks, 0, -1))]
        plan = {"schema_version": 1, "config": cfg, "settings": candidates(),
                "development": [asdict(Trial(101, 1101, 100, orders[0])), asdict(Trial(103, 1103, 200, orders[1]))],
                "test": [asdict(Trial(seed, seed + 2000, 700 + 100*i, orders[i % 2]))
                         for i, seed in enumerate((211, 223, 227, 229, 233, 239))],
                "utility_floor": .95, "meaningful_alignment_difference": .02,
                "interpretation": "bounded pilot; six independent model/sample/rule cells; uncertainty is across cells",
                "metadata": metadata()}
        write_json(out / "plan.json", plan)
        records = execute(out / "development.jsonl", cfg, plan["settings"], plan["development"])
        selection = select(records, plan["settings"], len(plan["development"]), plan["utility_floor"])
        write_json(out / "selection.json", {"plan_sha256": digest(plan), **selection})
        rows = []
        for setting in plan["settings"]:
            matches = [r for r in records if r["setting"] == setting]
            if len(matches) == len(plan["development"]) and all(r["status"] == "completed" for r in matches):
                rows.append({"group": setting["protocol"] + "/" + setting["method"], "setting": setting,
                             "metrics": aggregate([r["result"] for r in matches])})
        write_json(out / "development-summary.json", rows)
        plot(rows, out / "development-frontiers.png", frontier=True)
        groups = sorted({s["protocol"] + "/" + s["method"] for s in plan["settings"]})
        if groups != [s["group"] for s in selection["selected"]]:
            raise RuntimeError("a comparison group has no complete candidate; see development failures")
    else:
        plan = json.loads((out / "plan.json").read_text())
        selection = json.loads((out / "selection.json").read_text())
        if selection["plan_sha256"] != digest(plan):
            raise ValueError("plan changed after development")
        current_metadata = metadata()
        if current_metadata != plan["metadata"]:
            raise ValueError("runtime or source changed after development; start a separately labelled study")
        # Exclusive creation locks the selection before any held-out model is trained.
        with (out / "test-lock.json").open("x") as f:
            json.dump({"selection_sha256": digest(selection), "plan_sha256": digest(plan), "metadata": current_metadata}, f, indent=2)
        settings = [s["setting"] for s in selection["selected"]]
        records = execute(out / "test.jsonl", plan["config"], settings, plan["test"])
        rows = []
        for selected in selection["selected"]:
            matches = [r for r in records if r["setting"] == selected["setting"]]
            complete = len(matches) == len(plan["test"]) and all(r["status"] == "completed" for r in matches)
            rows.append({"group": selected["group"], "setting": selected["setting"], "complete": complete,
                         "utility_feasible_on_development": selected["utility_feasible_on_development"],
                         "metrics": aggregate([r["result"] for r in matches]) if complete else None})
        write_json(out / "test-summary.json", {"selection_sha256": digest(selection), "comparisons": rows})
        plot([r for r in rows if r["complete"]], out / "tradeoffs.png")
        if not all(r["complete"] for r in rows):
            raise RuntimeError("incomplete held-out comparisons; failed runs remain in test.jsonl")


if __name__ == "__main__":
    main()
