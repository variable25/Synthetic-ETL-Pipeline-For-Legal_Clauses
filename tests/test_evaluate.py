"""Tests for the shared scoring functions (offline, free)."""

import pytest

from src import config
from src.evaluate import compute_metrics, top_confusions

LABELS = list(config.LABELS)


def one_mistake():
    """Every label once; the Termination clause is wrongly predicted as Notices."""
    y_pred = LABELS.copy()
    y_pred[LABELS.index("Termination")] = "Notices"
    return LABELS, y_pred


def test_perfect_predictions_score_one():
    metrics = compute_metrics(LABELS, LABELS)
    assert metrics["macro_f1"] == 1.0
    assert metrics["accuracy"] == 1.0


def test_macro_f1_is_plain_average_over_all_labels():
    # 6 labels F1 = 1, Termination F1 = 0, Notices F1 = 2/3 (P = 1/2, R = 1)
    metrics = compute_metrics(*one_mistake())
    assert metrics["macro_f1"] == pytest.approx((6 + 0 + 2 / 3) / 8)
    assert metrics["accuracy"] == pytest.approx(7 / 8)


def test_never_predicted_label_counts_as_zero():
    metrics = compute_metrics(*one_mistake())
    assert metrics["per_class"]["Termination"]["f1"] == 0.0
    assert metrics["per_class"]["Termination"]["support"] == 1


def test_top_confusions_lists_the_mistake():
    assert top_confusions(compute_metrics(*one_mistake())) == [("Termination", "Notices", 1)]