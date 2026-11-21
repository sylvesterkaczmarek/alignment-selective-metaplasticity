from copy import deepcopy
from dataclasses import asdict
import gzip
import hashlib
from importlib.resources import files
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from alignment_metaplasticity.authority import score, strict_json
from alignment_metaplasticity.authority import main as scorer_cli
from alignment_metaplasticity.config import load_config
from alignment_metaplasticity.experiment import build_loaders, preflight_cells
from alignment_metaplasticity.importance import clone_parameters, normalize_importance
from alignment_metaplasticity.probes import intervention_probe
from alignment_metaplasticity.research_integrity import checked_metrics
from alignment_metaplasticity.study import Trial
from alignment_metaplasticity.tasks import BenchmarkSpec, make_alignment_tensors
from experiments.controlled_study import select

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = files("alignment_metaplasticity").joinpath("fixtures")


def fixture():
    manifest = json.loads(FIXTURES.joinpath("manifest.json").read_text())
    predictions = [
        json.loads(line)
        for line in FIXTURES.joinpath("predictions.jsonl").read_text().splitlines()
    ]
    return manifest, predictions


def test_demo_and_external_are_identical():
    manifest, predictions = fixture()
    expected = json.loads(FIXTURES.joinpath("expected.json").read_text())
    assert score(manifest, reversed(predictions)) == expected
    assert expected["runs"]["example"]["hidden_failure"] is True
    assert expected["runs"]["example"]["steps_below_floor_left_hold"] == 10


@pytest.mark.parametrize(
    "invalid",
    ["duplicate", "unknown-case", "unexpected-step", "unknown-action", "wrong-label"],
)
def test_invalid_predictions(invalid):
    manifest, pred = fixture()
    if invalid == "duplicate":
        pred.append(pred[0])
    else:
        key, value = {
            "unknown-case": ("pair", "missing"),
            "unexpected-step": ("checkpoint", True),
            "unknown-action": ("observed", "undecided"),
            "wrong-label": ("expected", "keep"),
        }[invalid]
        pred[0][key] = value
    with pytest.raises(ValueError):
        score(manifest, pred)


def test_missing_checkpoint_and_abstentions():
    m, p = fixture()
    p[0]["observed"] = None
    r = score(m, p[:-2])["runs"]["example"]
    assert (
        r["coverage"] == pytest.approx(2 / 3)
        and r["final_pass"] is None
        and r["hidden_failure"] is None
    )
    assert r["curve"][-1]["authorized"]["observed"] == 0
    assert r["curve"][0]["authorized"]["abstentions"] == 1
    assert r["baseline_pass"] is None and r["hidden_retention_failure"] is None
    assert r["curve"][0]["paired"] == {
        "correct": 0,
        "observed": 1,
        "expected": 1,
        "accuracy": 0,
    }
    assert r["curve"][-1]["paired"]["observed"] == 0


def test_initial_learning_and_later_loss_have_distinct_reports():
    manifest, pred = fixture()
    pred[0]["observed"] = "keep"
    pred[3]["observed"] = "keep"
    report = score(manifest, pred)["runs"]["example"]
    assert report["final_pass"] and report["hidden_failure"]
    assert report["baseline_pass"] is False
    assert report["hidden_retention_failure"] is None
    pred[0]["observed"] = "revise"
    assert score(manifest, pred)["runs"]["example"]["hidden_retention_failure"] is False
    pred[3]["observed"] = "revise"
    assert score(manifest, pred)["runs"]["example"]["hidden_retention_failure"] is True
    pred[-1]["observed"] = "revise"
    assert score(manifest, pred)["runs"]["example"]["hidden_retention_failure"] is False


def test_equal_marginals_can_hide_different_pair_accuracy():
    manifest, pred = fixture()
    manifest["pairs"].append(dict(manifest["pairs"][0], id="second"))
    pred += [dict(row, pair="second") for row in list(pred)]
    # At baseline and endpoint, one of two decisions per condition is correct.
    # Only at baseline are those correct decisions on the same pair.
    for row in pred:
        truth = manifest["pairs"][0][row["condition"]]
        correct = row["pair"] == "delivery"
        if row["checkpoint"] == 20 and row["condition"] == "unauthorized":
            correct = not correct
        row["observed"] = truth if correct else ({"keep", "revise"} - {truth}).pop()
    first, _, last = score(manifest, pred)["runs"]["example"]["curve"]
    for condition in ("authorized", "unauthorized"):
        assert first[condition]["accuracy"] == last[condition]["accuracy"] == 0.5
    assert first["paired"]["accuracy"] == 0.5 and last["paired"]["accuracy"] == 0


@pytest.mark.parametrize("name", ["manifest.json", "predictions.jsonl"])
@pytest.mark.parametrize("alias", ["same", "symbolic", "hard"])
def test_scorer_protects_source_inputs(tmp_path, monkeypatch, name, alias):
    for file in ("manifest.json", "predictions.jsonl"):
        (tmp_path / file).write_bytes(FIXTURES.joinpath(file).read_bytes())
    source = tmp_path / name
    output = source if alias == "same" else tmp_path / "report.json"
    if alias == "symbolic":
        output.symlink_to(source)
    if alias == "hard":
        os.link(source, output)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "asm-score",
            "score",
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--predictions",
            str(tmp_path / "predictions.jsonl"),
            "--out",
            str(output),
        ],
    )
    before = source.read_bytes()
    with pytest.raises(SystemExit) as error:
        scorer_cli()
    assert (
        error.value.code == 2
        and output.samefile(source)
        and source.read_bytes() == before
    )


@pytest.mark.parametrize("bad", ['{"a":1,"a":2}', '{"x":NaN}', '{"x":1e999}'])
def test_bad_json(bad):
    with pytest.raises(ValueError):
        strict_json(bad)


def test_fixture_works_without_rng_or_native_byte_order_assumptions():
    raw = FIXTURES.joinpath("canonical-tensors.json").read_bytes()
    assert (
        hashlib.sha256(raw).hexdigest()
        == "cd473f8ba18421946d0b249eb2dfdb43836e568d45ffb739817e43b6a1c9b874"
    )
    f = json.loads(raw)
    x, y = (
        torch.tensor(f["x"], dtype=torch.float32),
        torch.tensor(f["y"], dtype=torch.int64),
    )
    spec = BenchmarkSpec()
    equal = x[:4] == x[4:]
    assert not equal[:, spec.authorized_idx].any()
    equal[:, spec.authorized_idx] = True
    assert equal.all() and torch.equal(y[:4], 1 - y[4:])


def test_named_streams_remove_actual_cross_cell_overlap_and_preserve_legacy():
    cfg = load_config()
    spec = BenchmarkSpec()
    cfg["data"].update(
        alignment_train_size=128,
        alignment_eval_size=128,
        capability_train_size=64,
        capability_eval_size=64,
        bypass_train_size=64,
    )
    cells = [asdict(Trial(607, 7607, 4200)), asdict(Trial(617, 7617, 4400))]
    report = preflight_cells(cfg, cells, "controlled-v2")
    assert report["unique_signals"] > 0 and len(report["dataset_hashes"]) == 24
    with pytest.raises(ValueError, match="duplicate"):
        preflight_cells(cfg, [cells[0], cells[0]], "controlled-v2")

    def corpus(seed, ns, mode):
        return build_loaders(cfg, spec, seed, namespace=ns)[1][mode].dataset.tensors[0]

    assert torch.equal(
        corpus(7607, None, "unauthorized")[:, :8],
        corpus(7617, None, "authorized")[:, :8],
    )
    assert not torch.equal(
        corpus(7607, "controlled-v2", "unauthorized")[:, :8],
        corpus(7617, "controlled-v2", "authorized")[:, :8],
    )
    assert torch.equal(
        corpus(7607, None, "unauthorized"),
        make_alignment_tensors(128, spec, 7638, "unauthorized")[0],
    )


def test_overlap_changes_task_and_preserves_paired_nuisance_signal():
    spec = BenchmarkSpec()
    cfg = load_config()
    a = build_loaders(cfg, spec, 3)
    b = build_loaders(cfg, spec, 3, overlapping=True)
    x, y = a[2][1].dataset.tensors
    u, v = b[2][1].dataset.tensors
    assert torch.equal(x[:, :8], u[:, :8]) and not torch.equal(y, v)
    assert torch.count_nonzero(u[:, spec.proposal_idx]) == len(u)


def completed():
    path = ROOT / "results/calibrated-transfer-v2/test.jsonl"
    return next(
        r
        for line in path.read_text().splitlines()
        if (r := json.loads(line))["status"] == "completed"
    )


@pytest.mark.parametrize(
    "damage",
    [
        "harmonic",
        "history",
        "early",
        "acquisition",
        "order",
        "neutral",
        "removed-history",
    ],
)
def test_evidence_consistency(damage):
    r = completed()["result"]
    if damage == "harmonic":
        r["alignment"]["selective_corrigibility"] = 1.0
    elif damage == "history":
        r["history"][-1]["alignment"]["neutral_accuracy"] = 0.0
    elif damage == "early":
        r["earlier_retention"] = 0.01
    elif damage == "acquisition":
        r["acquisition"] = 0.02
    elif damage == "order":
        r["history"][0]["task"] = True
    elif damage == "neutral":
        r["initial_alignment"]["neutral_accuracy"] = 2.0
    else:
        del r["history"]
    with pytest.raises(ValueError):
        checked_metrics(r, require_history=True)


def test_all_archived_study_records_remain_valid():
    count = 0
    for folder in ("calibrated-transfer-v2", "controlled-v1"):
        for file in (ROOT / "results" / folder).glob("*.jsonl*"):
            with gzip.open(file, "rt") if file.suffix == ".gz" else file.open() as f:
                for line in f:
                    row = json.loads(line)
                    if row["status"] == "completed":
                        checked_metrics(row["result"], require_history=True)
                        count += 1
    assert count == 222


def test_studies_apply_their_declared_tie_breaks():
    a = completed()
    b = deepcopy(a)
    b["setting"]["lr"] = 0.03
    b["result"]["setting"]["lr"] = 0.03
    for row, acq in ((a, 0.91), (b, 0.93)):
        # Generic summaries are allowed outside the full-journal validation path.
        row["result"].pop("history")
        row["result"]["acquisition"] = acq
    settings = [a["setting"], b["setting"]]
    assert (
        select([a, b], settings, 1, 0.9, tie_break="candidate_order")["selected"][0][
            "setting"
        ]
        == settings[0]
    )
    assert (
        select([a, b], settings, 1, 0.9, tie_break="acquisition")["selected"][0][
            "setting"
        ]
        == settings[1]
    )


@pytest.mark.parametrize("broken", [False, True])
def test_frozen_backbone_restoration(broken):
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 3), torch.nn.Linear(3, 3), torch.nn.Linear(3, 2)
    )
    model[0].requires_grad_(False)
    model[2].requires_grad_(False)
    before = deepcopy(model.state_dict())
    calls = 0

    def measure():
        nonlocal calls
        calls += 1
        if calls == 2 and broken:
            raise RuntimeError("measurement")
        return {}

    run = SimpleNamespace(
        model=model,
        importance={n: v.abs() for n, v in clone_parameters(model).items()},
        alignment=measure,
        loaders=[None, None, None, {}],
        device=torch.device("cpu"),
    )
    if broken:
        with pytest.raises(RuntimeError, match="measurement"):
            intervention_probe(run)
    else:
        assert len(intervention_probe(run)["interventions"]) == 3
    assert all(torch.equal(before[n], v) for n, v in model.state_dict().items())


def test_float64_importance_and_half_precision_zeros():
    v = torch.tensor([1e100, 2e100], dtype=torch.float64)
    z = normalize_importance({"x": v}, 0.5)["x"]
    torch.testing.assert_close(z, torch.tensor([2 / 3, 1.0], dtype=torch.float64))
    assert torch.equal(
        normalize_importance({"w": torch.zeros(8, dtype=torch.float16)})["w"],
        torch.zeros(8),
    )


def test_packaged_two_family_evidence_and_missing_attempt():
    from scripts.verify_authority import verify
    from experiments.authority_analysis import validate, journal

    report = verify(ROOT)
    assert report["development_attempts"] == report["held_out_attempts"] == 168
    assert report["baseline_qualified_blocks"] == {"smol": 0, "qwen": 0}
    control = report["synthetic_control"]
    assert (
        control["attempts"]
        == control["completed"]
        == control["baseline_qualified"]
        == 84
    )
    assert control["hidden_retention_failures"] == 4
    assert {
        (e["condition"], e["method"], e["model_seed"]) for e in control["events"]
    } == {
        ("overlapping", "rehearsal", 801),
        ("overlapping", "ewc", 811),
        ("overlapping", "rehearsal", 823),
        ("overlapping", "rehearsal", 829),
    }
    directory = ROOT / "results/authority-v1"
    plan = json.loads((ROOT / "studies/authority-v1/protocol.json").read_text())
    lock = json.loads((directory / "lock.json").read_text())
    records = journal(directory / "test.jsonl.gz")
    with pytest.raises(ValueError, match="missing planned"):
        validate(plan, records[1:], "test", lock["blocks"], lock["choices"])
    records[0]["result"]["baseline_pass"] = not records[0]["result"]["baseline_pass"]
    with pytest.raises(ValueError, match="outcomes disagree"):
        validate(plan, records, "test", lock["blocks"], lock["choices"])


@pytest.mark.parametrize(
    "alias", ["test.jsonl.gz", "summary.json", "development.jsonl"]
)
def test_analysis_protects_published_evidence_before_loading(
    tmp_path, monkeypatch, alias
):
    from experiments.authority_analysis import main

    (tmp_path / "checksums.json").write_text("{}")
    monkeypatch.setattr(
        sys,
        "argv",
        ["authority_analysis", str(tmp_path), "--out", str(tmp_path / alias)],
    )
    with pytest.raises(ValueError, match="overwrite"):
        main()


@pytest.mark.parametrize("invalid", ["duplicate", "missing", "short", "boolean"])
def test_synthetic_recount_rejects_invalid_trials(invalid):
    from scripts.verify_authority import journal, recount_synthetic

    directory = ROOT / "results/authority-synthetic-v1"
    plan = json.loads((directory / "plan.json").read_text())
    rows = journal(directory / "runs.jsonl.gz")
    if invalid == "duplicate":
        rows[-1] = rows[0]
    elif invalid == "missing":
        rows.pop()
    elif invalid == "short":
        rows[0]["trajectory"].pop()
    else:
        rows[0]["trajectory"][0][0] = True
    with pytest.raises(ValueError, match="synthetic"):
        recount_synthetic(plan, rows, 0.9, 0.8)


def test_corrected_studies_are_complete_and_use_separate_streams():
    from alignment_metaplasticity.research_integrity import (
        read_journal,
        validate_records,
    )
    from scripts.verify_calibration import verify

    base = ROOT / "results/controlled-v2"
    plan = json.loads((base / "plan.json").read_text())
    assert plan["data_preflight"]["namespace"] == "controlled-v2"
    choices = json.loads((base / "selection.json").read_text())
    for name, settings in (
        ("development", plan["settings"]),
        ("test", [c["setting"] for c in choices["selected"]]),
    ):
        rows = read_journal(base / (name + ".jsonl.gz"))
        assert all(row["status"] == "completed" for row in rows)
        assert len(rows) == len(plan[name]) * len(settings)
        validate_records(rows, settings, plan[name], require_history=True)
    assert verify(ROOT / "results/calibrated-transfer-v3")["held_out_runs"] == 20
