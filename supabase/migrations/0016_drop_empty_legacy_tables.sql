-- NOT APPLIED YET, and optional. Paste into the Supabase SQL Editor to run it.
-- Claude's attempt on 2026-11-02 was stopped by a safety check on bulk deletes.

-- These seven tables belonged to Streamlit features that never held a row.
-- Nothing in the site or the n8n workflows reads them. The refusal below stops
-- this if any of them has data. Their definitions remain in init_database() in
-- legacy/property_analysis_dashboard.py, and the scoring formulas in
-- "Leading Indicators Modules" in the vault.
DO $$
DECLARE
  t text;
  n bigint;
BEGIN
  FOREACH t IN ARRAY ARRAY['market_commentary', 'market_sentiment', 'infrastructure_projects', 'migration_data', 'employment_data', 'employer_events', 'suburb_scores'] LOOP
    EXECUTE format('SELECT count(*) FROM public.%I', t) INTO n;
    IF n > 0 THEN
      RAISE EXCEPTION 'Table % has % rows, not dropping', t, n;
    END IF;
  END LOOP;
END $$;

DROP TABLE public.market_commentary;
DROP TABLE public.market_sentiment;
DROP TABLE public.infrastructure_projects;
DROP TABLE public.migration_data;
DROP TABLE public.employment_data;
DROP TABLE public.employer_events;
DROP TABLE public.suburb_scores;
