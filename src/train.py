"""
Fine-tuning: Legal-BERT learns the 8 clause labels from synthetic data only.

Trains on synthetic_train.jsonl, checks itself on synthetic_val.jsonl after
every epoch, and keeps the epoch with the lowest validation loss. LEDGAR is
never opened here, so the test set cannot influence training.

Run from the project root with:  python -m src.train
"""

import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from src import config
from src.evaluate import compute_metrics
from src.jsonl_io import read_jsonl

MODEL_ID = "nlpaueb/legal-bert-base-uncased"
OUTPUT_DIR = config.PROJECT_ROOT / "models" / "legal-bert-clauses"
MAX_TOKENS = 512        # BERT's limit; padding is per batch, so short clauses stay cheap
BATCH_SIZE = 16
LEARNING_RATE = 2e-5    # usual BERT fine-tuning range: 2e-5 to 5e-5
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.1   # first 10% of steps: learning rate ramps up from 0
MAX_EPOCHS = 8          # upper bound; early stopping usually ends sooner
PATIENCE = 2            # stop after this many epochs without a lower val loss
MIN_DELTA = 1e-3        # a val-loss drop smaller than this counts as "no improvement"
MAX_GRAD_NORM = 1.0     # clip gradients so one bad batch cannot wreck the weights

LABEL2ID = {label: i for i, label in enumerate(config.LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


# --- 1. Pure helpers --------------------------------------------------------------------
def encode_labels(labels: list[str]) -> list[int]:
    """Label names -> class ids. Stops loudly on a label we do not know."""
    unknown = set(labels) - set(LABEL2ID)
    if unknown:
        raise ValueError(f"Unknown label(s): {sorted(unknown)}")
    return [LABEL2ID[label] for label in labels]


class EarlyStopping:
    """Remembers the best validation loss and says when to give up."""

    def __init__(self, patience: int = PATIENCE, min_delta: float = MIN_DELTA):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.bad_epochs = 0

    def step(self, val_loss: float) -> bool:
        """Record one epoch. Returns True if it is clearly the best so far (= save it)."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.bad_epochs = 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience


# --- 2. Data ------------------------------------------------------------------------------
def make_loader(path: Path, tokenizer, shuffle: bool) -> DataLoader:
    """JSONL file -> batches of token ids + label ids, padded to the longest in each batch."""
    rows = read_jsonl(path)
    examples = list(zip([row["text"] for row in rows],
                        encode_labels([row["label"] for row in rows])))

    def collate(batch):
        texts, label_ids = zip(*batch)
        inputs = tokenizer(list(texts), truncation=True, max_length=MAX_TOKENS,
                           padding=True, return_tensors="pt")
        inputs["labels"] = torch.tensor(label_ids)
        return inputs

    generator = torch.Generator().manual_seed(config.RANDOM_SEED)  # same shuffle every run
    return DataLoader(examples, batch_size=BATCH_SIZE, shuffle=shuffle,
                      collate_fn=collate, generator=generator)


# --- 3. One pass over the data -------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, scheduler, device: str) -> float:
    """Update the weights once per batch. Returns the average training loss."""
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16,
                            enabled=device == "cuda"):
            loss = model(**batch).loss      # cross-entropy, computed by the model
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.inference_mode()
def evaluate_loader(model, loader, device: str) -> tuple[float, list[str], list[str]]:
    """No weight updates. Returns (average loss, true labels, predicted labels)."""
    model.eval()
    total_loss, y_true, y_pred = 0.0, [], []
    for batch in loader:
        batch = batch.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16,
                            enabled=device == "cuda"):
            output = model(**batch)
        total_loss += output.loss.item()
        y_pred += [ID2LABEL[i] for i in output.logits.argmax(dim=1).tolist()]
        y_true += [ID2LABEL[i] for i in batch["labels"].tolist()]
    return total_loss / len(loader), y_true, y_pred


# --- 4. Entry point -------------------------------------------------------------------------
def main() -> None:
    torch.manual_seed(config.RANDOM_SEED)   # same classifier-head init every run
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, id2label=ID2LABEL, label2id=LABEL2ID,
    ).to(device)

    train_loader = make_loader(config.SYNTHETIC_TRAIN_PATH, tokenizer, shuffle=True)
    val_loader = make_loader(config.SYNTHETIC_VAL_PATH, tokenizer, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    total_steps = MAX_EPOCHS * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(WARMUP_FRACTION * total_steps), total_steps)

    print(f"Fine-tuning {MODEL_ID} on {len(train_loader.dataset)} clauses "
          f"(val: {len(val_loader.dataset)}, {device})")
    stopper = EarlyStopping()
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        start = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, y_true, y_pred = evaluate_loader(model, val_loader, device)
        val_f1 = compute_metrics(y_true, y_pred)["macro_f1"]
        seconds = time.perf_counter() - start

        is_best = stopper.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "val_macro_f1": val_f1, "seconds": round(seconds)})
        print(f"epoch {epoch}  train_loss {train_loss:.4f}  val_loss {val_loss:.4f}  "
              f"val_macro_f1 {val_f1:.3f}  ({seconds:.0f}s)"
              + ("  <- best, saved" if is_best else ""))

        if is_best:
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
        if stopper.should_stop:
            print(f"No lower val loss for {PATIENCE} epochs: stopping.")
            break

    config.RESULTS_DIR.mkdir(exist_ok=True)
    history_path = config.RESULTS_DIR / "train_history.json"
    history_path.write_text(json.dumps({
        "model": MODEL_ID,
        "best_val_loss": stopper.best_loss,
        "hyperparameters": {"max_tokens": MAX_TOKENS, 
                            "batch_size": BATCH_SIZE,
                            "learning_rate": LEARNING_RATE, 
                            "weight_decay": WEIGHT_DECAY,
                            "warmup_fraction": WARMUP_FRACTION, 
                            "max_epochs": MAX_EPOCHS,
                            "patience": PATIENCE,
                            "min_delta": MIN_DELTA, 
                            "seed": config.RANDOM_SEED},
        "epochs": history,
    }, indent=2), encoding="utf-8")
    print(f"\nBest model -> {OUTPUT_DIR}\nHistory    -> {history_path}")


if __name__ == "__main__":
    main()