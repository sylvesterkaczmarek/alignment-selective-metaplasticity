"""Measure one frozen feature pass and one adapter in separate clean CPU processes."""

import argparse
import json
from pathlib import Path
import platform
import resource
import time

import torch
from alignment_metaplasticity.importance import normalize_importance
from experiments.authority_features import PLAN, encode, file_hash
from experiments.authority_data import make_block
from experiments.authority_study import Block


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("backbone", "adapter"))
    p.add_argument("family", choices=("smol", "qwen"))
    p.add_argument("--weights", type=Path)
    p.add_argument("--cache", type=Path)
    args = p.parse_args()
    plan = json.loads(PLAN.read_text())
    start = time.perf_counter()
    if args.phase == "backbone":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if args.weights is None:
            p.error("backbone requires --weights")
        for name, expected in plan["models"][args.family]["files"].items():
            if file_hash(args.weights / name) != expected:
                raise ValueError("unexpected model file")
        tok = AutoTokenizer.from_pretrained(args.weights, local_files_only=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.weights,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            use_safetensors=True,
            local_files_only=True,
        ).eval()
        model.requires_grad_(False)
        before = time.perf_counter()
        result = encode(model, tok, make_block("development", 0), 12)
        measured = {
            "feature_seconds": time.perf_counter() - before,
            "tokens": sum(result["tokens"]),
            "examples": len(result["tokens"]),
        }
    else:
        if args.cache is None:
            p.error("adapter requires --cache")
        b = Block(plan, args.family, "development", 0, args.cache)
        before = time.perf_counter()
        for _ in range(50):
            normalize_importance(b.omega)
        measured = {
            "mean_normalise_seconds": (time.perf_counter() - before) / 50,
            "adapter_parameters": b.identity["adapter_parameters"],
            "importance_bytes": sum(
                v.numel() * v.element_size() for v in b.omega.values()
            ),
        }
        measured["training_cost"] = b.run("fixed", plan["assignment_lr"])["cost"]
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(
        json.dumps(
            {
                "family": args.family,
                "phase": args.phase,
                "threads": torch.get_num_threads(),
                "elapsed_seconds": time.perf_counter() - start,
                "system": platform.system(),
                "lifetime_peak_rss_bytes": peak
                * (1 if platform.system() == "Darwin" else 1024),
                **measured,
            }
        )
    )


if __name__ == "__main__":
    main()
