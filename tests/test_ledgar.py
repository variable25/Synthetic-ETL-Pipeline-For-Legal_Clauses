"""Tests for the pure LEDGAR helpers (offline, no download)."""

import pytest

from src import config
from src.ledgar import LEDGAR_TO_OURS, check_label_map, filter_and_map

FAKE_NAMES = ["Adjustments", "Governing Laws", "Terminations"]


def test_map_covers_exactly_our_labels():
    assert sorted(LEDGAR_TO_OURS.values()) == sorted(config.LABELS)


def test_filter_keeps_renames_and_drops():
    rows = filter_and_map(["Laws  of\nNew York apply.", "Price adjusts.", "Either party may end it."],
                          [1, 0, 2], FAKE_NAMES)
    assert rows == [
        {"text": "Laws of New York apply.", "label": "Governing Law"},
        {"text": "Either party may end it.", "label": "Termination"},
    ]


def test_check_label_map_passes_when_all_present():
    check_label_map(list(LEDGAR_TO_OURS) + ["Adjustments"])  # must not raise


def test_check_label_map_raises_when_name_missing():
    names = [name for name in LEDGAR_TO_OURS if name != "Governing Laws"]
    with pytest.raises(ValueError, match="Governing Laws"):
        check_label_map(names)