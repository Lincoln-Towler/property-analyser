-- ALREADY APPLIED to production on 2026-11-04 (Supabase migration 20261104104414,
-- "cash_rate_decision_4_nov_2026"). Kept here as a record — do not re-run.

-- The RBA raised the cash rate by 0.25 to 4.85%, effective 4 Nov 2026 (rba.gov.au/statistics/cash-rate).
-- The feed only ran on Mondays, so it wouldn't have picked this up until 9 Nov. Same form as the
-- 30 Sep decision row. To undo: delete this one row.
insert into public.economic_indicators (date, indicator_name, value, source)
values ('2026-11-04', 'interest_rate', 4.85, 'RBA decision effective 4 Nov 2026 (+0.25)');
