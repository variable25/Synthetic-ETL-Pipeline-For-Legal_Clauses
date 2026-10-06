"""Tests for score.py's pure helper (offline, no model needed)."""

import pytest

from src import config
from src.evaluate import compute_metrics
from src.score import compare_reports

LABELS = list(config.LABELS)


def test_compare_reports_lists_every_label_then_macro_f1():
    perfect = compute_metrics(LABELS, LABELS)
    rows = compare_reports(perfect, perfect)
    assert [row[0] for row in rows] == LABELS + ["Macro-F1"]
    assert all(row[3] == 0 for row in rows)


def test_compare_reports_change_is_finetuned_minus_baseline():
    # baseline gets Termination wrong (F1 = 0), fine-tuned gets everything right
    wrong = LABELS.copy()
    wrong[LABELS.index("Termination")] = "Notices"
    baseline = compute_metrics(LABELS, wrong)
    finetuned = compute_metrics(LABELS, LABELS)

    rows = {row[0]: row for row in compare_reports(baseline, finetuned)}
    assert rows["Termination"][1:] == (0.0, 1.0, 1.0)
    assert rows["Macro-F1"][3] == pytest.approx(1.0 - baseline["macro_f1"])