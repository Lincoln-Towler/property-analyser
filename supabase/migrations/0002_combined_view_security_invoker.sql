-- 0002: Own the combined view in migrations, with SECURITY INVOKER.
--
-- Previously the Streamlit app DROP/CREATE'd this view at read time with
-- the owner connection, which made it SECURITY DEFINER (flagged by the
-- Supabase security advisor as an RLS bypass). Recreate it once, here,
-- with security_invoker so it enforces the querying role's RLS policies.
--
-- Semantics preserved from the app: rows from both tables, current table
-- (priority 1) beats history (priority 2) for the same (date, indicator_name).
-- Tiebreaker extended with id DESC so any future duplicate that slips in
-- resolves deterministically to the newest row instead of arbitrarily.

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
