-- 0003: Persist the daily market score.
--
-- The Streamlit app recomputed the score on every render and threw it
-- away — there was no way to see how the signal has moved over time.
-- The Next.js site writes one row per day (upsert) via a scheduled job;
-- the dashboard renders a score-history sparkline from it.

CREATE TABLE public.market_score_history (
    score_date   date PRIMARY KEY,
    final_score  numeric NOT NULL,
    signal       text NOT NULL,
    base_score   numeric,
    sub_scores   jsonb,
    breakdown    jsonb,
    confidence   jsonb,
    audit        jsonb,          -- per-indicator freshness/point-count snapshot
    commentary_md text,
    computed_at  timestamptz NOT NULL DEFAULT now()
);
