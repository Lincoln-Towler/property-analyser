-- Backfill template for a gappy or single-point indicator series.
--
-- WHY THIS EXISTS
-- Several scored indicators carry full weight off one stale reading, so they
-- hold the score in place instead of informing it. As of Aug 2026:
--   household_debt_gdp  1 reading, 2025-06-30, weight 25 (18.5% of the score)
--   mortgage_stress_rate 1 reading, 2026-03-24, weight 15 (11.1%)
--   credit_growth        missing 2026-03 and 2026-04
--   wage_growth          missing the 2025-12 quarter
-- With 3+ points in the window the engine's own trend and volatility
-- machinery starts working and the problem dissolves — no engine change.
--
-- ⚠️ PICK ONE SOURCE AND STAY WITH IT.
-- Published household-debt-to-GDP figures for Australia differ by up to 10
-- percentage points depending on methodology (IMF FSI ~110, Trading
-- Economics ~113.4, CEIC ~120.3 for the same quarter). That spread crosses
-- this project's own warning (110) and danger (120) thresholds, so mixing
-- sources inside one series will move the score for no real-world reason.
-- The existing 2025-06-30 = 113.7 row is labelled 'ABS'; match whatever
-- series that came from, and record the source on every row you add.
--
-- WHERE TO GET THE NUMBERS
--   household_debt_gdp   ABS 5232.0 Australian National Accounts: Finance
--                        and Wealth (quarterly), or RBA Chart Pack
--   mortgage_stress_rate the survey you originally used (Roy Morgan /
--                        Digital Finance Analytics publish differing series)
--   credit_growth        RBA D1 Growth in Financial Aggregates (monthly)
--   wage_growth          ABS 6345.0 Wage Price Index (quarterly)
--
-- HOW TO USE
-- Manual readings belong in economic_indicators (the n8n feed owns
-- economic_indicators_history). Edit the VALUES list and run in the
-- Supabase SQL editor. Safe to re-run: ON CONFLICT updates in place.
-- Migration 0001 constrains indicator_name and 0006 rejects implausible
-- values, so a typo or a bad figure fails loudly rather than landing.

INSERT INTO public.economic_indicators (date, indicator_name, value, source)
VALUES
    -- date        indicator             value   source (be specific — it shows on /audit)
    ('2024-09-30', 'household_debt_gdp',  000.0, 'ABS 5232.0 Sep-2024'),
    ('2024-12-31', 'household_debt_gdp',  000.0, 'ABS 5232.0 Dec-2024'),
    ('2025-03-31', 'household_debt_gdp',  000.0, 'ABS 5232.0 Mar-2025'),
    ('2025-09-30', 'household_debt_gdp',  000.0, 'ABS 5232.0 Sep-2025'),
    ('2025-12-31', 'household_debt_gdp',  000.0, 'ABS 5232.0 Dec-2025'),
    ('2026-03-31', 'household_debt_gdp',  000.0, 'ABS 5232.0 Mar-2026')
ON CONFLICT (date, indicator_name)
DO UPDATE SET value = EXCLUDED.value, source = EXCLUDED.source;

-- Check what landed:
-- SELECT date, value, source FROM economic_indicators_combined
-- WHERE indicator_name = 'household_debt_gdp' ORDER BY date;
--
-- Then reload /audit — the indicator should drop out of the
-- "riding on data that can't move" panel once it has 2+ recent readings.
