-- 0004: Enable Row Level Security everywhere.
--
-- !! PRE-FLIGHT (do NOT apply until verified):
-- !!  1. Confirm what credential n8n uses. A direct Postgres connection or
-- !!     service-role key bypasses RLS and keeps working. If n8n uses the
-- !!     ANON key via the REST API, add an insert policy for it first.
-- !!  2. Confirm nothing reads the btc_* tables (or get_btc_signal()) with
-- !!     the anon key. RLS-with-no-policies makes them invisible to anon.
-- !!  3. The Streamlit app connects as the table owner, so it keeps working
-- !!     during the parallel-run period (owners bypass RLS unless FORCEd).
--
-- End state:
--   * Public site reads via anon key: SELECT-only on indicator/property
--     tables and score history. No write policy exists for anon.
--   * Admin writes happen in Next.js Server Actions with the service-role
--     key (bypasses RLS), held only in server env vars.
--   * BTC tables: RLS on, no policies — untouched by anon/authenticated;
--     their service-role/owner writer is unaffected.

ALTER TABLE public.economic_indicators         ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.economic_indicators_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.property_data               ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.market_score_history        ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.market_sentiment            ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.market_commentary           ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.infrastructure_projects     ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.migration_data              ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.employment_data             ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.employer_events             ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.suburb_scores               ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.btc_key_levels              ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.btc_context_flags           ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.btc_signal_log              ENABLE ROW LEVEL SECURITY;

-- Read-only public access for the tables the website renders
CREATE POLICY public_read ON public.economic_indicators
    FOR SELECT TO anon, authenticated USING (true);
CREATE POLICY public_read ON public.economic_indicators_history
    FOR SELECT TO anon, authenticated USING (true);
CREATE POLICY public_read ON public.property_data
    FOR SELECT TO anon, authenticated USING (true);
CREATE POLICY public_read ON public.market_score_history
    FOR SELECT TO anon, authenticated USING (true);

-- Everything else: RLS on, zero policies (deny-all for anon/authenticated).
-- The empty legacy tables (market_sentiment, market_commentary,
-- infrastructure_projects, migration_data, employment_data,
-- employer_events, suburb_scores) can be DROPped at cutover instead —
-- kept locked-but-alive here so this migration is non-destructive.
