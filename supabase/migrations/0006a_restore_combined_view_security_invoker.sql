-- ALREADY APPLIED to production on 2026-08-08 (Supabase migration 20260808011749,
-- "restore_combined_view_security_invoker"). Added as a record on 2026-11-02 — do not re-run.

-- Restore the migration-owned combined view (the legacy Streamlit app's
-- runtime DDL recreated it as SECURITY DEFINER; the app is now patched to
-- stop doing that). Same definition as migration 0002.

DROP VIEW IF EXISTS public.economic_indicators_combined;

CREATE VIEW public.economic_indicators_combined
WITH (security_invoker = true) AS
SELECT date, indicator_name, value, source
FROM (
    SELECT date, indicator_name, value, source,
           ROW_NUMBER() OVER (
               PARTITION BY date, indicator_name
               ORDER BY priority ASC, id DESC
           ) AS rn
    FROM (
        SELECT id, date, indicator_name, value::numeric AS value, source, 1 AS priority
        FROM public.economic_indicators
        UNION ALL
        SELECT id, date, indicator_name, value, 'history' AS source, 2 AS priority
        FROM public.economic_indicators_history
    ) combined
) ranked
WHERE rn = 1;

GRANT SELECT ON public.economic_indicators_combined TO anon, authenticated;
