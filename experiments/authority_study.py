"""Development, locked held-out training and analysis for the authority benchmark."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import platform
import statistics
import time

import numpy as np
import torch
from torch import nn

from alignment_metaplasticity.authority import score
from alignment_metaplasticity.importance import (
    clone_parameters,
    compute_gradient_importance,
    ema_importance,
    plasticity_from_importance,
)
from alignment_metaplasticity.io_utils import write_json
from alignment_metaplasticity.streams import stream_seed
from alignment_metaplasticity.study import transform_plasticity, tensor_digest
from alignment_metaplasticity.tasks import make_loader
from alignment_metaplasticity.training import train_epochs
from .authority_data import NAMESPACE, digest, make_block, preflight
from .authority_features import PLAN, file_hash

CPU = torch.device("cpu")


class OutputAdapter(nn.Module):
    """Learn a bottleneck residual on the two native next-token logits."""

    def __init__(self, width, rank):
        super().__init__()
        self.down = nn.Linear(width, rank)
        self.up = nn.Linear(rank, 2, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return x[:, -2:] + self.up(self.down(x[:, :-2]).tanh())


def seed(phase, block, *coordinates):
    return stream_seed(NAMESPACE, phase, block, *coordinates)


class Block:
    def __init__(self, plan, family, phase, number, cache):
        self.plan, self.phase, self.number = plan, phase, number
        self.rows = make_block(phase, number)
        path = cache / f"{family}-{phase}-{number}.pt"
        data = torch.load(path, weights_only=True)
        meta = data["metadata"]
        if (
            meta["model"] != plan["models"][family]
            or meta["cases"] != digest(self.rows)
            or (meta["phase"], meta["block"]) != (phase, number)
            or meta["driver"]
            != file_hash(Path(__file__).with_name("authority_features.py"))
        ):
            raise ValueError("feature identity mismatch")
        features = data["features"]
        if (
            features.ndim != 2
            or features.shape[0] != len(self.rows)
            or data["logits"].shape != (len(self.rows), 2)
            or len(data["tokens"]) != len(self.rows)
        ):
            raise ValueError("invalid feature dimensions")
        normal = (
            features / features.square().mean(1, keepdim=True).clamp_min(1e-12).sqrt()
        )
        self.x = torch.cat([normal, data["logits"]], dim=1)
        if not torch.isfinite(self.x).all():
            raise ValueError("nonfinite features")
        self.y = torch.tensor([r["label"] for r in self.rows])
        self.groups = {
            (task, split): [
                i
                for i, r in enumerate(self.rows)
                if (r["task"], r["split"]) == (task, split)
            ]
            for task in ("authority", "schedule", "inspection")
            for split in ("train", "eval")
        }
        self.order = (
            ("schedule", "inspection")
            if number % 2 == 0
            else ("inspection", "schedule")
        )
        self.eval_rows = [self.rows[i] for i in self.groups["authority", "eval"]]
        pairs = [
            {"id": r["pair"], "authorized": "B", "unauthorized": "A"}
            for r in self.eval_rows
            if r["condition"] == "authorized"
        ]
        self.manifest = {
            "version": 1,
            "runs": ["adapter"],
            "actions": ["A", "B"],
            "pairs": pairs,
            "checkpoints": list(range(2 * plan["epochs_per_task"] + 1)),
            "failure_floor": plan["failure_floor"],
            "release_floor": plan["release_floor"],
        }
        torch.manual_seed(seed(phase, number, "initialization"))
        self.model = OutputAdapter(features.shape[1], plan["rank"])
        train_epochs(
            self.model,
            self.loader("authority", "alignment"),
            CPU,
            plan["pretrain_epochs"],
            plan["pretrain_lr"],
            0.0,
        )
        self.anchor = clone_parameters(self.model)
        self.omega = self.estimate(self.model)
        self.identity = {
            "initial_weights": tensor_digest(self.anchor),
            "corpus": meta["cases"],
            "features": tensor_digest({"x": self.x, "y": self.y}),
            "adapter_parameters": sum(p.numel() for p in self.model.parameters()),
            "backbone_tokens": sum(data["tokens"]),
            "feature_seconds": data["seconds"],
            "feature_runtime": {
                k: meta[k]
                for k in ("torch", "transformers", "dtype", "threads", "batch_size")
            },
        }

    def loader(self, task, purpose):
        ids = self.groups[task, "train"]
        return make_loader(
            (self.x[ids], self.y[ids]),
            self.plan["batch_size"],
            purpose != "importance",
            seed(self.phase, self.number, task, purpose),
        )

    def estimate(self, model):
        return compute_gradient_importance(
            model,
            self.loader("authority", "importance"),
            CPU,
            max_batches=9999,
            quantile=0.95,
        )

    @torch.no_grad()
    def decisions(self, model, task):
        return model(self.x[self.groups[task, "eval"]]).argmax(1)

    def run(self, method, lr):
        p = self.plan
        start = time.perf_counter()
        model = copy.deepcopy(self.model)
        omega = self.omega
        trace, utility = [], []

        def measure():
            decisions = self.decisions(model, "authority").tolist()
            trace.append("".join("AB"[v] for v in decisions))

        measure()
        for task in self.order:
            args = {}
            if method in ("fixed", "refreshed", "uniform", "shuffled"):
                coef = plasticity_from_importance(
                    omega, p["importance_strength"], p["min_plasticity"]
                )
                kind = method if method in ("uniform", "shuffled") else "metaplastic"
                args["grad_scale"] = transform_plasticity(
                    coef, kind, seed(self.phase, self.number, "permutation")
                )
            elif method == "ewc":
                args.update(
                    ewc_anchor=self.anchor,
                    ewc_importance=omega,
                    ewc_lambda=p["ewc_lambda"],
                )

            def checkpoint(epoch):
                if method == "rehearsal" and epoch == p["epochs_per_task"]:
                    train_epochs(
                        model,
                        self.loader("authority", "rehearsal-" + task),
                        CPU,
                        1,
                        lr,
                        0.0,
                    )
                measure()

            train_epochs(
                model,
                self.loader(task, "capability"),
                CPU,
                p["epochs_per_task"],
                lr,
                0.0,
                epoch_callback=checkpoint,
                **args,
            )
            if method == "refreshed" and task == self.order[0]:
                omega = ema_importance(omega, self.estimate(model), 0.5)
            utility.append(
                {
                    t: (self.decisions(model, t) == self.y[self.groups[t, "eval"]])
                    .double()
                    .mean()
                    .item()
                    for t in self.order
                }
            )
        predictions = (
            {
                "run": "adapter",
                "checkpoint": i,
                "pair": r["pair"],
                "condition": r["condition"],
                "observed": letter,
            }
            for i, letters in enumerate(trace)
            for r, letter in zip(self.eval_rows, letters)
        )
        report = score(self.manifest, predictions)["runs"]["adapter"]
        curve = report["curve"]
        initial = curve[0]["minimum_component"] >= p["release_floor"]
        auth_n = len(self.groups["authority", "train"])
        cap_n = (
            sum(len(self.groups[t, "train"]) for t in self.order) * p["epochs_per_task"]
        )
        importance_passes = {
            "fixed": 1,
            "ewc": 1,
            "refreshed": 2,
            "uniform": 1,
            "shuffled": 1,
        }.get(method, 0)
        rehearsal_n = 2 * auth_n if method == "rehearsal" else 0
        return {
            "trace": trace,
            "capabilities": utility,
            "acquisition": statistics.mean(v[t] for v, t in zip(utility, self.order)),
            "earlier_retention": utility[-1][self.order[0]],
            "baseline_pass": initial,
            "final_pass": report["final_pass"],
            "hidden_failure": initial and report["hidden_failure"],
            "trough": min(x["minimum_component"] for x in curve),
            "below_floor": report["below_floor_checkpoints"],
            "steps_below": report["steps_below_floor_left_hold"],
            "cost": {
                "pretrain_examples": p["pretrain_epochs"] * auth_n,
                "capability_examples": cap_n,
                "rehearsal_examples": rehearsal_n,
                "importance_examples": auth_n * importance_passes,
                "adaptation_steps": sum(
                    math.ceil(len(self.groups[t, "train"]) / p["batch_size"])
                    for t in self.order
                )
                * p["epochs_per_task"]
                + (2 * math.ceil(auth_n / p["batch_size"]) if rehearsal_n else 0),
                "seconds": time.perf_counter() - start,
            },
            "final_weights": tensor_digest(clone_parameters(model)),
        }


def train_phase(plan, phase, count, cache, path, choices=None):
    with path.open("x", encoding="utf8") as f:
        for family in plan["models"]:
            for block in range(count):
                try:
                    obj = Block(plan, family, phase, block, cache)
                    setup_error = None
                except (ValueError, FloatingPointError, RuntimeError) as exc:
                    setup_error = f"{type(exc).__name__}: {exc}"
                for method in plan["methods"]:
                    rates = (
                        plan["learning_rates"]
                        if choices is None
                        else [choices[family][method]["lr"]]
                    )
                    if choices and method in ("fixed", "shuffled"):
                        rates = list(dict.fromkeys([*rates, plan["assignment_lr"]]))
                    for lr in rates:
                        row = {
                            "family": family,
                            "phase": phase,
                            "block": block,
                            "method": method,
                            "lr": lr,
                        }
                        try:
                            if setup_error is not None:
                                raise ValueError(setup_error)
                            row.update(
                                status="completed",
                                identity=obj.identity,
                                result=obj.run(method, lr),
                            )
                        except (ValueError, FloatingPointError, RuntimeError) as exc:
                            row.update(
                                status="failed", error=f"{type(exc).__name__}: {exc}"
                            )
                        f.write(
                            json.dumps(row, allow_nan=False, separators=(",", ":"))
                            + "\n"
                        )
                        f.flush()
                print(phase, family, block, flush=True)


def read_rows(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def selection(plan, rows):
    choices, variances = {}, []
    for family in plan["models"]:
        choices[family] = {}
        chosen = {}
        for method in plan["methods"]:
            options = []
            for lr in plan["learning_rates"]:
                records = [
                    r
                    for r in rows
                    if (r["family"], r["method"], r["lr"]) == (family, method, lr)
                ]
                if sorted(r["block"] for r in records) != list(
                    range(plan["development_blocks"])
                ) or any(r["status"] != "completed" for r in records):
                    continue
                results = [
                    r["result"] for r in sorted(records, key=lambda r: r["block"])
                ]
                low = min(
                    statistics.mean(r[k] for r in results)
                    for k in ("acquisition", "earlier_retention")
                )
                passed = low >= plan["utility_floor"]
                options.append(
                    (
                        (
                            passed,
                            statistics.mean(r["trough"] for r in results)
                            if passed
                            else low,
                        ),
                        lr,
                        results,
                    )
                )
            if not options:
                raise ValueError(
                    f"no complete development candidate for {family}/{method}"
                )
            (feasible, _), lr, results = max(options, key=lambda item: item[0])
            choices[family][method] = {"lr": lr, "feasible": feasible}
            chosen[method] = results
        for control in ("shuffled", "uniform", "ewc"):
            variances.append(
                statistics.variance(
                    int(r["hidden_failure"]) - int(s["hidden_failure"])
                    for r, s in zip(chosen["fixed"], chosen[control])
                )
            )
    target = math.ceil(1.96**2 * max(0.10, *variances) / 0.20**2)
    return {
        "choices": choices,
        "uncapped_blocks": target,
        "blocks": max(plan["min_test_blocks"], min(plan["max_test_blocks"], target)),
    }


def source_id():
    root = PLAN.parents[2]
    paths = [
        *sorted((root / "src/alignment_metaplasticity").glob("*.py")),
        *sorted((root / "experiments").glob("authority_*.py")),
        PLAN,
    ]
    return digest({str(p.relative_to(root)): file_hash(p) for p in paths})


def lock(plan, out):
    path = out / "development.jsonl"
    chosen = selection(plan, read_rows(path))
    blocks = [("development", i) for i in range(plan["development_blocks"])]
    blocks += [("test", i) for i in range(chosen["blocks"])]
    return {
        **chosen,
        "protocol": digest(plan),
        "development_sha256": file_hash(path),
        "source": source_id(),
        "corpus_preflight": preflight(blocks),
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "machine": platform.machine(),
            "threads": torch.get_num_threads(),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("development", "test"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    p = json.loads(PLAN.read_text())
    torch.use_deterministic_algorithms(True)
    if args.phase == "development":
        args.out.mkdir(parents=True, exist_ok=False)
        preflight([("development", i) for i in range(p["development_blocks"])])
        train_phase(
            p,
            "development",
            p["development_blocks"],
            args.cache,
            args.out / "development.jsonl",
        )
        write_json(args.out / "lock.json", lock(p, args.out))
    else:
        frozen = json.loads((args.out / "lock.json").read_text())
        if frozen != lock(p, args.out):
            raise ValueError(
                "source, protocol, runtime, data or selection changed after development"
            )
        train_phase(
            p,
            "test",
            frozen["blocks"],
            args.cache,
            args.out / "test.jsonl",
            frozen["choices"],
        )


if __name__ == "__main__":
    main()
