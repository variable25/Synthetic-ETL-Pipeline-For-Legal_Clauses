"""Tests for export.py's pure helpers (offline, no database)."""

from collections import Counter

from src import config
from src.export import remove_test_overlap, split_train_val
from src.validate import compute_hash


def make_samples(per_label: int) -> list[dict]:
    """per_label fake samples for each of the 8 labels."""
    return [{"sample_id": f"{label}-{i}", "label": label, "text": f"{label} clause {i}"}
            for label in config.LABELS for i in range(per_label)]


def test_overlap_removal_ignores_case_and_whitespace():
    samples = [{"text": "The laws of Ohio govern."}, {"text": "Notices must be in writing."}]
    test_hashes = {compute_hash("the  LAWS of ohio\ngovern.")}
    kept, n_removed = remove_test_overlap(samples, test_hashes)
    assert kept == [{"text": "Notices must be in writing."}]
    assert n_removed == 1


def test_split_is_stratified():
    train, val = split_train_val(make_samples(20), val_fraction=0.1, seed=42)
    val_counts = Counter(s["label"] for s in val)
    assert all(val_counts[label] == 2 for label in config.LABELS)
    assert len(train) == 144


def test_split_loses_nothing_and_has_no_overlap():
    samples = make_samples(20)
    train, val = split_train_val(samples, val_fraction=0.1, seed=42)
    train_ids = {s["sample_id"] for s in train}
    val_ids = {s["sample_id"] for s in val}
    assert not train_ids & val_ids
    assert train_ids | val_ids == {s["sample_id"] for s in samples}


def test_split_is_reproducible_with_same_seed():
    first = split_train_val(make_samples(20), val_fraction=0.1, seed=42)
    second = split_train_val(make_samples(20), val_fraction=0.1, seed=42)
    assert first == second