from __future__ import annotations
import json
import os
from pathlib import Path
import tempfile
from typing import Any


def require_distinct_output(output: Path, *inputs: Path) -> None:
    """Reject output paths that identify a protected input, including aliases."""
    destination = output.resolve()
    for source in inputs:
        if destination == source.resolve() or (
            output.exists() and source.exists() and output.samefile(source)
        ):
            raise ValueError(f"output would overwrite input or evidence file: {source}")


def write_json(path: str | Path, payload: Any) -> Path:
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path
