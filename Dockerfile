# --- Stage 1: build the React frontend ------------------------------------------
FROM node:24-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# --- Stage 2: Python runtime (CPU only) --------------------------------------------
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    MODEL_DIR=/app/models/legal-bert-clauses \
    STATIC_DIR=/app/frontend/dist
WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY models/legal-bert-clauses/ models/legal-bert-clauses/
COPY serve/ serve/
COPY --from=frontend /app/frontend/dist frontend/dist

RUN useradd --create-home appuser
USER appuser

EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn serve.app:app --host 0.0.0.0 --port ${PORT:-8080} --forwarded-allow-ips='*'"]
