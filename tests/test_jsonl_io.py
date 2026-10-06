"""Round-trip test for the shared JSONL helpers."""

from src.jsonl_io import read_jsonl, write_jsonl


def test_round_trip_keeps_rows_and_special_characters(tmp_path):
    rows = [{"text": "Section § 5 applies to the café.", "label": "Notices"},
            {"text": "Either party may terminate.", "label": "Termination"}]
    path = tmp_path / "subfolder" / "rows.jsonl"
    write_jsonl(rows, path)
    assert read_jsonl(path) == rows