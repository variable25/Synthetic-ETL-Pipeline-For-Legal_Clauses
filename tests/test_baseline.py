"""Tests for the baseline's pure helper (offline, no model download)."""

from src import config
from src.baseline import HYPOTHESIS_TEMPLATE, build_pairs


def test_build_pairs_is_clause_major():
    premises, hypotheses = build_pairs(["clause A", "clause B"])
    n = len(config.LABELS)
    assert premises == ["clause A"] * n + ["clause B"] * n
    assert hypotheses[:n] == [HYPOTHESIS_TEMPLATE.format(label) for label in config.LABELS]
    assert hypotheses[n:] == hypotheses[:n]