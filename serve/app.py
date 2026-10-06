"""
Inference API: classifies one contract clause with the fine-tuned Legal-BERT.

Self-contained on purpose: imports nothing from src/, so the container needs
no OpenAI key, no Postgres and none of the training libraries.

Run locally from the project root with:  uvicorn serve.app:app --reload
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import torch
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, StringConstraints
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", PROJECT_ROOT / "models" / "legal-bert-clauses"))
STATIC_DIR = Path(os.getenv("STATIC_DIR", PROJECT_ROOT / "frontend" / "dist"))
MAX_TOKENS = 512        # same limit as training
MAX_CHARS = 5000        # reject huge inputs before they reach the model


# --- 1. Request / response shapes ---------------------------------------------------
ClauseText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1,
                                              max_length=MAX_CHARS)]


class ClassifyRequest(BaseModel):
    text: ClauseText


class LabelScore(BaseModel):
    label: str
    score: float


class ClassifyResponse(BaseModel):
    label: str                  # the top prediction
    scores: list[LabelScore]    # every label, highest first
    truncated: bool             # True if the clause was longer than MAX_TOKENS


# --- 2. Pure helper -------------------------------------------------------------------
def rank_scores(labels: list[str], probs: list[float]) -> list[LabelScore]:
    """Pair each label with its probability, highest first."""
    scores = [LabelScore(label=label, score=p) for label, p in zip(labels, probs, strict=True)]
    return sorted(scores, key=lambda s: s.score, reverse=True)


# --- 3. The model ---------------------------------------------------------------------
class Classifier:
    """Loads the fine-tuned model once and classifies single clauses on CPU."""

    def __init__(self, model_dir: Path):
        if not model_dir.exists():
            raise RuntimeError(f"Model not found at {model_dir}. Run: python -m src.train")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
        id2label = self.model.config.id2label
        self.labels = [id2label[i] for i in range(len(id2label))]

    @torch.inference_mode()
    def classify(self, text: str) -> tuple[list[float], bool]:
        """Probability per label (in self.labels order) and whether the text was cut off."""
        truncated = len(self.tokenizer(text)["input_ids"]) > MAX_TOKENS
        inputs = self.tokenizer(text, truncation=True, max_length=MAX_TOKENS,
                                return_tensors="pt")
        logits = self.model(**inputs).logits[0]
        return torch.softmax(logits, dim=-1).tolist(), truncated


# --- 4. The app -----------------------------------------------------------------------
def create_app(classifier=None) -> FastAPI:
    """Build the app. Tests pass a fake classifier; production loads the real one."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.classifier = classifier if classifier is not None else Classifier(MODEL_DIR)
        yield

    app = FastAPI(title="Legal clause classifier", lifespan=lifespan)

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/api/classify", response_model=ClassifyResponse)
    def classify(request: ClassifyRequest) -> ClassifyResponse:
        model = app.state.classifier
        probs, truncated = model.classify(request.text)
        scores = rank_scores(model.labels, probs)
        return ClassifyResponse(label=scores[0].label, scores=scores, truncated=truncated)

    if STATIC_DIR.exists():     # the built React app, once it exists (step 4.2)
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="frontend")
    return app


app = create_app()