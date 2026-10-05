-- schema.sql
-- Creates the three tables for the synthetic data ETL pipeline.
-- Runs automatically on the FIRST start of the Postgres container
-- (mounted into /docker-entrypoint-initdb.d/ by docker-compose.yml).

-- 1. Parent: one row per pipeline run (lineage + cost tracking)
CREATE TABLE IF NOT EXISTS generation_runs (
    run_id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    started_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at       TIMESTAMPTZ,
    status            TEXT NOT NULL DEFAULT 'running'
                      CHECK (status IN ('running', 'completed', 'budget_stopped', 'failed')),
    model             TEXT NOT NULL,
    prompt_version    TEXT NOT NULL,
    target_samples    INTEGER NOT NULL CHECK (target_samples > 0),
    n_accepted        INTEGER NOT NULL DEFAULT 0,
    n_rejected        INTEGER NOT NULL DEFAULT 0,
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    cost_eur          NUMERIC(10, 6) NOT NULL DEFAULT 0
);

-- 2. Child: accepted, validated training samples
CREATE TABLE IF NOT EXISTS synthetic_samples (
    sample_id     BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id        BIGINT NOT NULL
                  REFERENCES generation_runs (run_id) ON DELETE CASCADE,
    label         TEXT NOT NULL CHECK (label IN (
                      'Governing Law', 'Termination', 'Confidentiality', 'Indemnification',
                      'Notices', 'Severability', 'Assignment', 'Entire Agreement')),
    clause_text   TEXT NOT NULL,
    text_hash     CHAR(64) NOT NULL UNIQUE,
    contract_type TEXT NOT NULL,
    tone          TEXT NOT NULL CHECK (tone IN ('formal', 'plain', 'legalese')),
    length_bucket TEXT NOT NULL CHECK (length_bucket IN ('short', 'medium', 'long')),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 3. Child: rejected generations, kept for debugging and quality metrics
CREATE TABLE IF NOT EXISTS rejected_samples (
    reject_id  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id     BIGINT NOT NULL
               REFERENCES generation_runs (run_id) ON DELETE CASCADE,
    raw_output TEXT NOT NULL,
    reason     TEXT NOT NULL
               CHECK (reason IN ('invalid_json', 'too_short', 'wrong_label', 'duplicate')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 4. Indexes for frequent lookups (UNIQUE on text_hash already has its own)
CREATE INDEX IF NOT EXISTS idx_samples_run_id  ON synthetic_samples (run_id);
CREATE INDEX IF NOT EXISTS idx_samples_label   ON synthetic_samples (label);
CREATE INDEX IF NOT EXISTS idx_rejected_run_id ON rejected_samples (run_id);