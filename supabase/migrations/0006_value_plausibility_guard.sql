-- 0006: Value plausibility guard.
--
-- Background (Aug 2026): the n8n interest_rate scraper broke mid-July and
-- wrote literal 0 every week (rows 127-130, deleted before this migration).
-- The unique constraint stops duplicate rows but nothing stopped garbage
-- values. Ranges are deliberately generous — they reject "the parser
-- returned nothing", not unusual-but-real readings.
--
-- Enforcement differs by table on purpose:
--   * economic_indicators (manual entry): hard CHECK -> a typo'd value
--     errors loudly in Supabase Studio / the admin form.
--   * economic_indicators_history (n8n feed): BEFORE INSERT trigger that
--     silently SKIPS implausible rows -> a broken scraper can't poison the
--     series, and can't fail a batched insert that carries other
--     indicators. The gap it leaves shows up as staleness on /audit.

CREATE OR REPLACE FUNCTION public.is_plausible_value(name text, v numeric)
RETURNS boolean
LANGUAGE sql IMMUTABLE
SET search_path = ''
AS $$
  SELECT CASE name
    WHEN 'interest_rate'          THEN v >= 0.01 AND v <= 25
    WHEN 'household_debt_gdp'     THEN v >= 20   AND v <= 300
    WHEN 'rental_vacancy_rate'    THEN v >= 0.05 AND v <= 25
    WHEN 'building_approvals'     THEN v >= 1000 AND v <= 500000  -- monthly OR annual units pass
    WHEN 'mortgage_stress_rate'   THEN v >= 0.5  AND v <= 90
    WHEN 'unemployment_rate'      THEN v >= 0.5  AND v <= 30
    WHEN 'auction_clearance_rate' THEN v >= 10   AND v <= 100
    WHEN 'mortgage_arrears_rate'  THEN v >= 0.01 AND v <= 20
    -- credit_growth / wage_growth / population_growth / dwelling_supply_deficit
    -- can legitimately be zero or negative -> no range enforced
    ELSE true
  END;
$$;

ALTER TABLE public.economic_indicators
  ADD CONSTRAINT economic_indicators_plausible_value_check
  CHECK (public.is_plausible_value(indicator_name, value::numeric));

CREATE OR REPLACE FUNCTION public.skip_implausible_history_rows()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = ''
AS $$
BEGIN
  IF NOT public.is_plausible_value(NEW.indicator_name, NEW.value) THEN
    RAISE WARNING 'skipping implausible % value % on %', NEW.indicator_name, NEW.value, NEW.date;
    RETURN NULL;  -- drop the row, keep the rest of the batch
  END IF;
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS history_plausibility_guard ON public.economic_indicators_history;
CREATE TRIGGER history_plausibility_guard
  BEFORE INSERT OR UPDATE ON public.economic_indicators_history
  FOR EACH ROW EXECUTE FUNCTION public.skip_implausible_history_rows();
