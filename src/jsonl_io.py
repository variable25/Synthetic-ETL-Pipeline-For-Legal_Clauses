"""
Reading and writing JSONL files (one JSON object per line).

Shared by every module that touches data/, so all files are written
and read the same way.
"""

import json
from pathlib import Path


def write_jsonl(rows: list[dict], path: Path) -> None:
    """Write one JSON object per line, creating the folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    """Read a file written by write_jsonl back into a list of dicts."""
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]