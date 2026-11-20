"""Reproducible CPU streams, with effective 32-bit collision checks."""

from __future__ import annotations

import hashlib
import json


def stream_seed(namespace: str, *coordinates: str | int) -> int:
    identity = json.dumps(
        [namespace, *coordinates], separators=(",", ":"), ensure_ascii=True
    )
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4], "big")


def dataset_streams(namespace, sample_seed, tasks):
    purposes = [
        "alignment/train",
        "alignment/authorized",
        "alignment/unauthorized",
        "alignment/neutral",
        "challenge/attack",
        "challenge/preserve",
        "challenge/eval",
    ]
    purposes += [
        f"capability/{task}/{split}"
        for task in range(1, tasks + 1)
        for split in ("train", "eval")
    ]
    return {
        f"{purpose}/{kind}": stream_seed(namespace, sample_seed, purpose, kind)
        for purpose in purposes
        for kind in ("examples", "loader")
    }
