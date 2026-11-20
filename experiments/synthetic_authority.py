"""Paired CPU control with shared authority/capability features and epoch sampling."""

import argparse
from functools import partial
import json
from pathlib import Path


from alignment_metaplasticity.config import load_config
from alignment_metaplasticity.evaluation import accuracy
from alignment_metaplasticity.experiment import build_loaders, preflight_cells
from alignment_metaplasticity.io_utils import write_json
from alignment_metaplasticity.streams import stream_seed
from alignment_metaplasticity.study import StudyRun, Setting, Trial
from alignment_metaplasticity.tasks import make_alignment_tensors, make_loader

NAMESPACE = "authority-synthetic-v1"


class EpochRun(StudyRun):
    def train(self, loader, epochs, phase, *, lr=None, **kwargs):
        if phase == "capability":
            if not hasattr(self, "trajectory"):
                self.trajectory = []
                paired_seed = stream_seed(
                    NAMESPACE, self.trial.sample_seed, "counterfactual-pairs"
                )
                self.pairs = [
                    make_loader(
                        make_alignment_tensors(
                            128, self.spec, paired_seed, mode=condition
                        ),
                        128,
                        False,
                        paired_seed,
                    )
                    for condition in ("authorized", "unauthorized")
                ]
                self.observe()
            kwargs["epoch_callback"] = lambda epoch: self.observe()
        result = super().train(loader, epochs, phase, lr=lr, **kwargs)
        if phase == "rehearsal":
            self.observe()
        return result

    def observe(self):
        self.trajectory.append(
            [accuracy(self.model, loader, self.device) for loader in self.pairs]
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cfg = load_config(
        overrides={
            "model": {"hidden_dim": 32, "depth": 1, "num_capability_tasks": 2},
            "training": {"alignment_epochs": 30, "capability_epochs": 6},
            "data": {
                "alignment_train_size": 384,
                "alignment_eval_size": 256,
                "capability_train_size": 384,
                "capability_eval_size": 256,
                "batch_size": 64,
            },
        }
    )
    trials = [
        {
            "model_seed": x,
            "sample_seed": x + 20000,
            "rule_seed": x + 30000,
            "order": [1, 2] if i % 2 == 0 else [2, 1],
        }
        for i, x in enumerate((801, 811, 823, 827, 829, 839))
    ]
    methods = [
        ("fine_tune", "fixed"),
        ("metaplastic", "fixed"),
        ("metaplastic", "periodic"),
        ("shuffled", "fixed"),
        ("uniform", "fixed"),
        ("ewc", "fixed"),
        ("rehearsal", "periodic"),
    ]
    plan = {
        "namespace": NAMESPACE,
        "config": cfg,
        "trials": trials,
        "methods": methods,
        "conditions": ["separate", "overlapping"],
        "preflight": preflight_cells(cfg, trials, NAMESPACE),
        "cadence": "after pretraining and after each capability epoch; rehearsal adds a recorded post-rehearsal point",
    }
    write_json(args.out / "plan.json", plan)
    with (args.out / "runs.jsonl").open("x") as f:
        for condition in plan["conditions"]:
            for trial in trials:
                for method, protocol in methods:
                    row = {
                        "condition": condition,
                        "trial": trial,
                        "method": method,
                        "protocol": protocol,
                    }
                    try:
                        run = EpochRun(
                            cfg,
                            Trial(**trial),
                            Setting(method, protocol, ewc_lambda=1.0),
                            loader_factory=partial(
                                build_loaders,
                                namespace=NAMESPACE,
                                overlapping=condition == "overlapping",
                            ),
                        )
                        result = run.run()
                        row.update(
                            status="completed",
                            trajectory=run.trajectory,
                            acquisition=result["acquisition"],
                            earlier_retention=result["earlier_retention"],
                            initial_weights=result["initial_digest"],
                            final_weights=result["final_digest"],
                            data=result["data_sha256"],
                        )
                    except (ValueError, RuntimeError, FloatingPointError) as exc:
                        row.update(
                            status="failed", error=f"{type(exc).__name__}: {exc}"
                        )
                    f.write(
                        json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n"
                    )
                    f.flush()


if __name__ == "__main__":
    main()
