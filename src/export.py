"""
Export step: Postgres -> train/validation JSONL files for train.py.

First drops any synthetic clause that also appears in the LEDGAR test set
(so the exam never leaks into training), then makes a stratified, seeded
train/validation split.

Run from the project root with:  python -m src.export
"""

from collections import Counter

from sklearn.model_selection import train_test_split

from src import config, db
from src.jsonl_io import read_jsonl, write_jsonl
from src.validate import compute_hash


# --- 1. Pure helpers ----------------------------------------------------------------
def remove_test_overlap(samples: list[dict],
                        test_hashes: set[str]) -> tuple[list[dict], int]:
    """Drop samples whose text matches a test clause (same normalized hash)."""
    kept = [s for s in samples if compute_hash(s["text"]) not in test_hashes]
    return kept, len(samples) - len(kept)


def split_train_val(samples: list[dict], val_fraction: float = config.VAL_FRACTION,
                    seed: int = config.RANDOM_SEED) -> tuple[list[dict], list[dict]]:
    """Stratified split: every label keeps the same share in train and val."""
    labels = [s["label"] for s in samples]
    train, val = train_test_split(samples, test_size=val_fraction,
                                  stratify=labels, random_state=seed)
    return train, val


# --- 2. Entry point -------------------------------------------------------------------
def main() -> None:
    if not config.LEDGAR_TEST_PATH.exists():
        raise SystemExit(f"{config.LEDGAR_TEST_PATH} not found. Run: python -m src.ledgar")

    with db.get_connection() as conn:
        samples = db.fetch_samples(conn)
    n_fetched = len(samples)
    test_hashes = {compute_hash(row["text"]) for row in read_jsonl(config.LEDGAR_TEST_PATH)}

    samples, n_overlap = remove_test_overlap(samples, test_hashes)
    train, val = split_train_val(samples)
    write_jsonl(train, config.SYNTHETIC_TRAIN_PATH)
    write_jsonl(val, config.SYNTHETIC_VAL_PATH)

    print(f"Fetched {n_fetched} samples from Postgres")
    print(f"Removed {n_overlap} that also appear in the LEDGAR test set")
    print(f"Train: {len(train)} -> {config.SYNTHETIC_TRAIN_PATH}")
    print(f"Val:   {len(val)} -> {config.SYNTHETIC_VAL_PATH}\n")

    train_counts = Counter(s["label"] for s in train)
    val_counts = Counter(s["label"] for s in val)
    print(f"{'label':<17} {'train':>6} {'val':>5}")
    for label in config.LABELS:
        print(f"{label:<17} {train_counts[label]:>6} {val_counts[label]:>5}")


if __name__ == "__main__":
    main()