"""Tests for the inference API (offline: a fake classifier replaces Legal-BERT)."""

import pytest
from fastapi.testclient import TestClient

from serve.app import MAX_CHARS, create_app, rank_scores


class FakeClassifier:
    labels = ["Notices", "Termination", "Assignment"]

    def classify(self, text):
        return [0.2, 0.7, 0.1], len(text) > 100


@pytest.fixture
def client():
    with TestClient(create_app(classifier=FakeClassifier())) as client:
        yield client


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_classify_returns_top_label_and_all_scores_sorted(client):
    body = client.post("/api/classify", json={"text": "Notice shall be given in writing."}).json()
    assert body["label"] == "Termination"
    assert [s["label"] for s in body["scores"]] == ["Termination", "Notices", "Assignment"]
    assert body["truncated"] is False


def test_classify_reports_truncation(client):
    body = client.post("/api/classify", json={"text": "x " * 100}).json()
    assert body["truncated"] is True


@pytest.mark.parametrize("text", ["", "   "])
def test_classify_rejects_empty_text(client, text):
    assert client.post("/api/classify", json={"text": text}).status_code == 422


def test_classify_rejects_too_long_text(client):
    assert client.post("/api/classify", json={"text": "a" * (MAX_CHARS + 1)}).status_code == 422


def test_rank_scores_sorts_highest_first():
    ranked = rank_scores(["A", "B", "C"], [0.1, 0.6, 0.3])
    assert [(s.label, s.score) for s in ranked] == [("B", 0.6), ("C", 0.3), ("A", 0.1)]