"""
Real evaluation data: the LEDGAR test split, filtered to our 8 labels.

LEDGAR (part of the LexGLUE benchmark) holds clauses from real SEC contracts,
labelled by humans. It is used ONLY for the final evaluation, never for
training or tuning, so the "no hand labels" claim stays true.

Run from the project root with:  python -m src.ledgar
"""

import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset

from src.validate import normalize_text

DATASET_ID = "coastalcph/lex_glue"
DATASET_CONFIG = "ledgar"
SPLIT = "test"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "ledgar_test.jsonl"


# --- 1. LEDGAR label name -> our label name (exact counterparts only) -----------
LEDGAR_TO_OURS = {
    "Governing Laws": "Governing Law",
    "Terminations": "Termination",
    "Confidentiality": "Confidentiality",
    "Indemnifications": "Indemnification",
    "Notices": "Notices",
    "Severability": "Severability",
    "Assignments": "Assignment",
    "Entire Agreements": "Entire Agreement",
}


# --- 2. Pure helpers ----------------------------------------------------------------
def check_label_map(ledgar_names: list[str]) -> None:
    """Stop loudly if a name we map from does not exist in the dataset."""
    missing = set(LEDGAR_TO_OURS) - set(ledgar_names)
    if missing:
        raise ValueError(f"LEDGAR has no label(s) named: {sorted(missing)}")


def filter_and_map(texts: list[str], label_ids: list[int],
                   ledgar_names: list[str]) -> list[dict]:
    """Keep only clauses with one of our labels, renamed to our label names."""
    rows = []
    for text, label_id in zip(texts, label_ids, strict=True):
        our_label = LEDGAR_TO_OURS.get(ledgar_names[label_id])
        if our_label is not None:
            rows.append({"text": normalize_text(text), "label": our_label})
    return rows


# --- 3. Output ------------------------------------------------------------------------
def write_jsonl(rows: list[dict], path: Path) -> None:
    """Write one JSON object per line, creating the folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    split = load_dataset(DATASET_ID, DATASET_CONFIG, split=SPLIT)
    ledgar_names = split.features["label"].names
    check_label_map(ledgar_names)

    rows = filter_and_map(split["text"], split["label"], ledgar_names)
    write_jsonl(rows, OUTPUT_PATH)

    print(f"Kept {len(rows)} of {len(split)} LEDGAR {SPLIT} clauses -> {OUTPUT_PATH}")
    for label, count in sorted(Counter(row["label"] for row in rows).items()):
        print(f"  {label:<17} {count}")


if __name__ == "__main__":
    main()