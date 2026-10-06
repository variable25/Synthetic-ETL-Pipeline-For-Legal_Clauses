-- stats.sql
-- Read-only summary of the generation runs, used for the README.
-- Run with:  docker exec -i synthetic_etl_db sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"' < sql/stats.sql

-- 1. One row per pipeline run
SELECT run_id, status, prompt_version, target_samples, n_accepted, n_rejected,
       prompt_tokens, completion_tokens, round(cost_eur, 4) AS cost_eur,
       date_trunc('second', finished_at - started_at) AS duration
FROM generation_runs
ORDER BY run_id;

-- 2. Totals across all runs
SELECT sum(n_accepted) AS accepted, sum(n_rejected) AS rejected,
       round(100.0 * sum(n_accepted) / nullif(sum(n_accepted) + sum(n_rejected), 0), 1)
           AS acceptance_pct,
       sum(prompt_tokens) AS prompt_tokens, sum(completion_tokens) AS completion_tokens,
       round(sum(cost_eur), 4) AS cost_eur
FROM generation_runs;

-- 3. Why generations were rejected
SELECT reason, count(*) AS n
FROM rejected_samples
GROUP BY reason
ORDER BY n DESC;

-- 4. Accepted samples per label (should be balanced)
SELECT label, count(*) AS n
FROM synthetic_samples
GROUP BY label
ORDER BY label;

-- 5. Recipe-grid coverage: tone x length
SELECT tone, length_bucket, count(*) AS n
FROM synthetic_samples
GROUP BY tone, length_bucket
ORDER BY tone, length_bucket;