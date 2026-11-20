"""Cache exact frozen final-token features for the two-action adapter."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import torch

from .authority_data import digest, make_block

PLAN = Path(__file__).resolve().parents[1] / "studies/authority-v1/protocol.json"


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def encode(model, tokenizer, cases, batch_size):
    letters = [tokenizer.encode(a, add_special_tokens=False) for a in ("A", "B")]
    if any(len(ids) != 1 for ids in letters):
        raise ValueError("each response must be a single native token")
    selected = model.get_output_embeddings().weight[[v[0] for v in letters]].detach().float()
    features, logits, tokens = [], [], []
    with torch.inference_mode():
        for start in range(0, len(cases), batch_size):
            prompts = [tokenizer.apply_chat_template([
                {"role": "system", "content": "Answer with the single letter A or B."},
                {"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True)
                for r in cases[start:start + batch_size]]
            pack = tokenizer(prompts, add_special_tokens=False, padding=True, return_tensors="pt")
            lengths = pack["attention_mask"].sum(1)
            hidden = model.base_model(**pack, use_cache=False).last_hidden_state
            h = hidden[torch.arange(len(prompts)), lengths - 1].float()
            features.append(h)
            logits.append(h @ selected.T)
            tokens.extend(lengths.tolist())
    return {"features": torch.cat(features), "logits": torch.cat(logits), "tokens": tokens}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("smol", "qwen"))
    parser.add_argument("phase", choices=("development", "test"))
    parser.add_argument("--blocks", type=int, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--batch-size", type=int, default=12)
    args = parser.parse_args()
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import snapshot_download
    desc = json.loads(PLAN.read_text())["models"][args.model]
    weights = args.weights or Path(snapshot_download(desc["id"], revision=desc["revision"],
                                  allow_patterns=list(desc["files"])))
    for name, expected in desc["files"].items():
        if file_hash(weights/name) != expected:
            raise ValueError(f"model artifact differs from pinned revision: {name}")
    tokenizer = AutoTokenizer.from_pretrained(weights, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(weights, local_files_only=True, trust_remote_code=False,
                use_safetensors=True, dtype=torch.bfloat16, attn_implementation="eager").eval().requires_grad_(False)
    args.cache.mkdir(parents=True, exist_ok=True)
    for block in range(args.blocks):
        cases = make_block(args.phase, block)
        path = args.cache / f"{args.model}-{args.phase}-{block}.pt"
        meta = {"model": desc, "phase": args.phase, "block": block, "cases": digest(cases),
                "driver": file_hash(Path(__file__)), "torch": str(torch.__version__),
                "transformers": transformers.__version__, "dtype": "bfloat16", "batch_size": args.batch_size,
                "threads": torch.get_num_threads()}
        if path.exists():
            if torch.load(path, weights_only=True)["metadata"] != meta:
                raise ValueError("feature cache belongs to a different protocol/runtime")
            continue
        tick = time.monotonic()
        result = encode(model, tokenizer, cases, args.batch_size)
        result.update(metadata=meta, seconds=time.monotonic()-tick)
        fd, temp = tempfile.mkstemp(dir=args.cache)
        os.close(fd)
        try:
            torch.save(result, temp)
            os.replace(temp, path)
        finally:
            Path(temp).unlink(missing_ok=True)
        print(path.name, round(result["seconds"], 2), flush=True)


if __name__ == "__main__":
    main()
