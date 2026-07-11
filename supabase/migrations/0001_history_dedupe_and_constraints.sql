-- 0001: Repair economic_indicators_history and guard indicator names.
--
-- The history table (fed by n8n) has no unique constraint, so workflow
-- re-runs appended pure duplicates (verified June 2026: every duplicate
-- group has exactly one distinct value, e.g. 18 identical wage_growth
-- rows dated 2025-09-01). Dedupe keeping the earliest row, then make
-- duplicates impossible and switch n8n's INSERT to an upsert.
--
-- !! Take a manual pg_dump before applying — the free tier has no
-- !! automated backups.

BEGIN;

-- 1. Remove pure duplicates, keeping the first-inserted row per (date, indicator_name)
DELETE FROM public.economic_indicators_history h
USING public.economic_indicators_history keep
WHERE keep.date = h.date
  AND keep.indicator_name = h.indicator_name
  AND keep.id < h.id;

-- 2. Never again
ALTER TABLE public.economic_indicators_history
  ADD CONSTRAINT economic_indicators_history_date_indicator_key
  UNIQUE (date, indicator_name);

-- 3. Guard against typo'd indicator names creating phantom series.
--    List = INDICATORS_CONFIG keys + the non-scored extras that exist in data.
--    Applied to both tables; extend the list when a new indicator is added
--    (keep in sync with web/src/lib/scoring/config.ts).
CREATE OR REPLACE FUNCTION public.is_known_indicator(name text)
RETURNS boolean
LANGUAGE sql IMMUTABLE
SET search_path = ''
AS $$
  SELECT name = ANY (ARRAY[
    'interest_rate',
    'household_debt_gdp',
    'rental_vacancy_rate',
    'building_approvals',
    'mortgage_stress_rate',
    'unemployment_rate',
    'auction_clearance_rate',
    'credit_growth',
    'wage_growth',
    'mortgage_arrears_rate',
    'dwelling_supply_deficit',
    'population_growth'
  ]);
$$;

ALTER TABLE public.economic_indicators
  ADD CONSTRAINT economic_indicators_known_name_check
  CHECK (public.is_known_indicator(indicator_name)) NOT VALID;

ALTER TABLE public.economic_indicators_history
  ADD CONSTRAINT economic_indicators_history_known_name_check
  CHECK (public.is_known_indicator(indicator_name)) NOT VALID;

-- Validate separately so pre-existing rows are checked explicitly;
-- if this fails, inspect with:
--   SELECT DISTINCT indicator_name FROM economic_indicators_history
--   WHERE NOT public.is_known_indicator(indicator_name);
ALTER TABLE public.economic_indicators VALIDATE CONSTRAINT economic_indicators_known_name_check;
ALTER TABLE public.economic_indicators_history VALIDATE CONSTRAINT economic_indicators_history_known_name_check;

COMMIT;

-- n8n follow-up (manual, in the n8n workflow): change the insert node to
--   INSERT INTO economic_indicators_history (date, indicator_name, value)
--   VALUES (...)
--   ON CONFLICT (date, indicator_name) DO UPDATE SET value = EXCLUDED.value;
