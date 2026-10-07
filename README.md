cat > README.md <<'EOF'
# Synthetic ETL Pipeline for Legal Clause Classification

**€0.23 of LLM-generated training data, zero hand labels: a fine-tuned Legal-BERT reaches 0.956 macro-F1 on 2,424 real contract clauses, up from 0.843 for a zero-shot baseline.**

**Live demo:** https://clause-classifier-tc75lmsdvq-ey.a.run.app (Cloud Run, scales to zero, so the first request after idle can take about 30 s)

| | Zero-shot baseline | Fine-tuned on synthetic data |
|---|---|---|
| Model | `facebook/bart-large-mnli` | `nlpaueb/legal-bert-base-uncased` |
| Training data | none | 5,064 synthetic clauses (€0.23, 0 hand labels) |
| Macro-F1 on LEDGAR test | 0.843 | **0.956** |
| Accuracy on LEDGAR test | 0.854 | **0.969** |

A teacher LLM (gpt-4o-mini) writes labelled contract clauses. An ETL pipeline validates them and loads them into Postgres. A small student model (Legal-BERT) is fine-tuned on that data alone and then scored on real clauses from SEC filings that it never saw during training or tuning. The model is served by FastAPI with a React frontend, packaged in Docker and deployed on Google Cloud Run.

---

## Architecture

```mermaid
flowchart LR
    A["Recipe grid<br/>8 labels × 8 contract types<br/>× 3 tones × 3 lengths"] --> B["Extract<br/>gpt-4o-mini, JSON mode<br/>10 clauses per call"]
    B --> C["Transform<br/>validate.py: JSON shape, label,<br/>min length, SHA-256 dedup"]
    C -->|accepted| D[("Load: Postgres<br/>synthetic_samples")]
    C -->|rejected + reason| E[("rejected_samples")]
    D --> F["export.py<br/>LEDGAR overlap guard<br/>stratified 90/10 split"]
    F --> G["train.py<br/>fine-tune Legal-BERT"]
    G --> H["score.py<br/>LEDGAR test set"]
    L["LEDGAR test split<br/>2,424 real clauses"] --> F
    L --> H