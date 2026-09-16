"""Atomic JSON writes. A power cut mid-write must not leave a truncated result file."""
from __future__ import annotations
import json, os
from pathlib import Path


def atomic_write_json(path, obj, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=indent, default=float)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
