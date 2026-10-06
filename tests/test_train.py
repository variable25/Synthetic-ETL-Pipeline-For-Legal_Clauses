"""Tests for train.py's pure helpers (offline, no model download)."""

import pytest

from src import config
from src.train import ID2LABEL, EarlyStopping, encode_labels


def test_encode_labels_follows_config_order():
    assert encode_labels(list(config.LABELS)) == list(range(len(config.LABELS)))
    assert [ID2LABEL[i] for i in encode_labels(["Notices", "Assignment"])] == [
        "Notices", "Assignment"]


def test_encode_labels_rejects_unknown_label():
    with pytest.raises(ValueError, match="Warranty"):
        encode_labels(["Notices", "Warranty"])


def test_early_stopping_saves_only_on_improvement():
    stopper = EarlyStopping(patience=2)
    assert stopper.step(0.50) is True
    assert stopper.step(0.40) is True
    assert stopper.step(0.45) is False
    assert stopper.best_loss == 0.40


def test_early_stopping_stops_after_patience_and_resets_on_improvement():
    stopper = EarlyStopping(patience=2)
    stopper.step(0.40)
    stopper.step(0.45)
    assert not stopper.should_stop
    stopper.step(0.30)              # improvement resets the counter
    stopper.step(0.31)
    assert not stopper.should_stop
    stopper.step(0.32)
    assert stopper.should_stop

def test_early_stopping_ignores_tiny_improvements():
    stopper = EarlyStopping(patience=2, min_delta=1e-3)
    stopper.step(0.0009)
    assert stopper.step(0.0007) is False      # 0.0002 lower: noise, not progress
    assert stopper.bad_epochs == 1